"""
Load and preprocess recordings.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, Iterable, Tuple, Union
from os import path, scandir
from functools import partial
import numpy as np  # type: ignore
from librosa.util import normalize  # type: ignore
from librosa.feature import melspectrogram  # type: ignore
import tensorflow as tf  # type: ignore
# pylint: disable=redefined-outer-scope

DATA_DIR = 'recordings'
ACCENTS = sorted(acc.name for acc in scandir(DATA_DIR))
# The output mel shape depends on librosa default parameters
MEL_SHAPE = (40, 128)


def file_split(accent: str, hyp: Dict[str, Union[float, int]]) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load filenames into datasets and split into train and test."""
    acc_dir = path.join(DATA_DIR, accent)
    acc_glob = path.join(acc_dir, '*.wav')
    file_list = tf.data.Dataset.list_files(acc_glob)
    num_files = sum(1 for _ in scandir(acc_dir))
    num_test = round(hyp['test_split'] * num_files)
    return file_list.skip(num_test), file_list.take(num_test)


def mel_generator(data: tf.data.Dataset,
                  hyp: Dict[str, Union[float, int]]) -> Iterable[tf.Tensor]:
    """Construct mel spec generator for given dataset."""
    for audio in data:
        spec = melspectrogram(np.array(audio), sr=hyp['sampling_rate'])
        mel_log = normalize(np.log(spec + 1e-9))
        # transpose to get time on axis 0
        yield tf.convert_to_tensor(mel_log.T, dtype=tf.float32)


def transform_files(filenames: tf.data.Dataset,
                    label: int,
                    hyp: Dict[str, Union[float, int]]) -> tf.data.Dataset:
    """Transform filenames of the same label to labelled dataset."""
    audio = filenames.map(tf.io.read_file) \
        .map(tf.audio.decode_wav) \
        .map(lambda tup: tup[0]) \
        .map(lambda amp: tf.reshape(amp, [-1])) \
        .map(lambda sig: tf.signal.frame(sig, hyp['frame_len'],
                                         hyp['frame_step'])) \
        .interleave(tf.data.Dataset.from_tensor_slices,
                    cycle_length=tf.data.AUTOTUNE,
                    num_parallel_calls=tf.data.AUTOTUNE)
    specs = tf.data.Dataset.from_generator(
        partial(mel_generator, audio, hyp),
        output_signature=tf.TensorSpec(shape=MEL_SHAPE))
    lbl = [0.0] * len(ACCENTS)
    lbl[label] = 1.0
    const = tf.data.Dataset.from_tensors(lbl).repeat()
    return tf.data.Dataset.zip((specs, const))


def load_accents(hyp: Dict[str, Union[float, int]]) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load all accents into datasets."""
    train, test = [transform_files(ds, 0, hyp)
                   for ds in file_split(ACCENTS[0], hyp)]
    for lbl, accent in enumerate(ACCENTS[1:], start=1):
        current_train, current_test = [transform_files(ds, lbl, hyp)
                                       for ds in file_split(accent, hyp)]
        train = train.concatenate(current_train)
        test = test.concatenate(current_test)
    return train.shuffle(hyp['shuffle_buffer']), \
        test.shuffle(hyp['shuffle_buffer'])
