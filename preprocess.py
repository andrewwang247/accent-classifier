"""
Load and preprocess recordings.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, List, Tuple, Union
from os import path
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import tensorflow_io as tfio  # type: ignore
from util import DATA_DIR, ACCENTS
# pylint: disable=redefined-outer-name


def _file_split(accent: str, hyp: Dict[str, Union[float, int]]) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load filenames into datasets and split into train, val, test."""
    acc_dir = path.join(DATA_DIR, accent)
    acc_glob = path.join(acc_dir, '*.wav')
    file_list = tf.data.Dataset.list_files(acc_glob)
    num_files = sum(1 for _ in file_list)
    num_test = round(hyp['test_split'] * num_files)
    num_val = round(hyp['val_split'] * num_files)
    test = file_list.take(num_test)
    train_val = file_list.skip(num_test)
    val = train_val.take(num_val)
    train = train_val.skip(num_val)
    return train, val, test


def _standardize_tensor(tens: tf.Tensor) -> tf.Tensor:
    """Rescale tensor to 0 mean and 1 std."""
    mean = tf.reduce_mean(tens)
    std = tf.math.reduce_std(tens)
    denom = std if std != 0.0 else tf.constant(1e-4)
    return (tens - mean) / denom


def _decode_wav_files(filenames: tf.data.Dataset) -> tf.data.Dataset:
    """Read, decode, and reshape filenames dataset."""
    return filenames.map(tf.io.read_file) \
        .map(tf.audio.decode_wav) \
        .map(lambda amp: tf.reshape(amp[0], [-1]))


def dataset_class_weights(accents: List[str]) -> Dict[int, float]:
    """Compute class weights on imbalanced accents."""
    samples = np.empty(len(accents), dtype=int)
    for idx, accent in enumerate(accents):
        glb = path.join(DATA_DIR, accent, '*.wav')
        filenames = tf.data.Dataset.list_files(glb)
        audio = _decode_wav_files(filenames)
        samples[idx] = sum(aud.shape[0] for aud in audio)
    weights = np.sum(samples) / samples / len(accents)
    return dict(enumerate(weights / np.sum(weights)))


def _frame_audio(audio: tf.data.Dataset,
                 hyp: Dict[str, Union[float, int]]) -> tf.data.Dataset:
    """Split audio into frames."""
    return audio.map(lambda sig: tf.signal.frame(sig, hyp['frame_len'],
                                                 hyp['frame_step'])) \
        .interleave(tf.data.Dataset.from_tensor_slices,
                    cycle_length=tf.data.AUTOTUNE,
                    num_parallel_calls=tf.data.AUTOTUNE)


def _transform_files(filenames: tf.data.Dataset,
                     label: int,
                     hyp: Dict[str, Union[float, int]]) -> tf.data.Dataset:
    """Transform filenames of the same label to labelled dataset."""
    audio = _decode_wav_files(filenames)
    frames = _frame_audio(audio, hyp)
    specs = frames.map(lambda sig: tfio.experimental.audio.
                       spectrogram(sig, hyp['num_fft'],
                                   hyp['spec_window'],
                                   hyp['spec_stride'])) \
        .map(lambda spec: tfio.experimental.audio
             .melscale(spec, hyp['sampling_rate'], hyp['num_mels'],
                       hyp['freq_min'], hyp['freq_max'])) \
        .map(_standardize_tensor)
    const = tf.data.Dataset.from_tensors([label]).repeat()
    return tf.data.Dataset.zip((specs, const))


def load_accents(hyp: Dict[str, Union[float, int]]) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load all accents into datasets."""
    train, val, test = [_transform_files(ds, 0, hyp)
                        for ds in _file_split(ACCENTS[0], hyp)]
    for lbl, accent in enumerate(ACCENTS[1:], start=1):
        new_train, new_val, new_test = [_transform_files(ds, lbl, hyp)
                                        for ds in _file_split(accent, hyp)]
        train = train.concatenate(new_train)
        val = val.concatenate(new_val)
        test = test.concatenate(new_test)
    return train, val, test
