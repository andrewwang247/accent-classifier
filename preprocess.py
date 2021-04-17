"""
Load and preprocess recordings.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, Tuple, Union
from os import path
from functools import partial
import tensorflow as tf  # type: ignore
import tensorflow_io as tfio  # type: ignore
from pydub import AudioSegment  # type: ignore
from pydub.silence import split_on_silence  # type: ignore
from util import DATA_DIR, ACCENTS
from util import _standardize_tensor, _normalize_tensor
# pylint: disable=redefined-outer-name


def _file_split(accent: str, hyp: Dict[str, Union[float, int]]) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load filenames into datasets and split into train, val, test."""
    acc_glob = path.join(DATA_DIR, accent, '*.wav')
    file_list = tf.data.Dataset.list_files(acc_glob)
    num_files = sum(1 for _ in file_list)
    num_test = round(hyp['test_split'] * num_files)
    num_val = round(hyp['val_split'] * num_files)
    test = file_list.take(num_test)
    train_val = file_list.skip(num_test)
    val = train_val.take(num_val)
    train = train_val.skip(num_val)
    return train, val, test


def _remove_silence(fpath: tf.Tensor,
                    hyp: Dict[str, Union[float, int]]) \
        -> tf.Tensor:
    """Load and remove silence silence from audio file."""
    audio = AudioSegment.from_wav(bytes.decode(fpath.numpy(), 'utf-8'))
    chunks = split_on_silence(audio,
                              min_silence_len=hyp['min_silence_len'],
                              silence_thresh=hyp['silence_thresh'])
    combined = sum(chunks)
    assert isinstance(combined, AudioSegment)
    sample = combined.get_array_of_samples()
    return tf.convert_to_tensor(sample, dtype=tf.int16)


def _transform_files(filenames: tf.data.Dataset,
                     label: int,
                     hyp: Dict[str, Union[float, int]]) -> tf.data.Dataset:
    """Transform filenames of the same label to labelled dataset."""
    clean_up = partial(_remove_silence, hyp=hyp)
    py_func = tf.autograph.experimental.do_not_convert(
        lambda fpath: tf.py_function(clean_up, [fpath], tf.int16))
    frames = filenames.map(py_func) \
        .map(_normalize_tensor) \
        .map(lambda sig: tf.signal.frame(sig, hyp['frame_len'],
                                         hyp['frame_step'])) \
        .interleave(tf.data.Dataset.from_tensor_slices,
                    cycle_length=tf.data.AUTOTUNE,
                    num_parallel_calls=tf.data.AUTOTUNE)
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
