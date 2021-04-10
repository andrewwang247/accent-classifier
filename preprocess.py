"""
Load and preprocess recordings.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, Tuple, Union
from os import path, scandir
import tensorflow as tf  # type: ignore
# pylint: disable=redefined-outer-name

DATA_DIR = 'recordings'
ACCENTS = sorted(acc.name for acc in scandir(DATA_DIR))


def file_split(accent: str, hyp: Dict[str, Union[float, int]]) \
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
    const = tf.data.Dataset.from_tensors([label]).repeat()
    return tf.data.Dataset.zip((audio, const))


def load_accents(hyp: Dict[str, Union[float, int]]) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load all accents into datasets."""
    train, val, test = [transform_files(ds, 0, hyp)
                        for ds in file_split(ACCENTS[0], hyp)]
    for lbl, accent in enumerate(ACCENTS[1:], start=1):
        new_train, new_val, new_test = [transform_files(ds, lbl, hyp)
                                        for ds in file_split(accent, hyp)]
        train = train.concatenate(new_train)
        val = val.concatenate(new_val)
        test = test.concatenate(new_test)
    buff = hyp['shuffle_buffer']
    return train.shuffle(buff), \
        val.shuffle(buff), \
        test.shuffle(buff)
