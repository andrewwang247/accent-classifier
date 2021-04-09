"""
Load and preprocess recordings.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, Iterable, Tuple, Union
from os import path, listdir
from functools import partial
import numpy as np  # type: ignore
from librosa.util import normalize  # type: ignore
from librosa.feature import melspectrogram  # type: ignore
import tensorflow as tf  # type: ignore

DATA_DIR = 'recordings'


def _mel_frame(frame: tf.Tensor) -> tf.Tensor:
    """Compute the mel spectrogram of frame."""
    mel = melspectrogram(frame.numpy(), sr=8000)
    mel_log = normalize(np.log(mel + 1e-9))
    return tf.convert_to_tensor(mel_log, dtype=tf.float32)


def _frame_accent(accent: str, label: int) \
        -> Iterable[Tuple[tf.Tensor, tf.Tensor]]:
    """Generate labeled frames from every audio of given accent."""
    # TODO: modify these to change shape of data
    frame_len, step = 30000, 2000
    wav_glob = path.join(DATA_DIR, accent, '*.wav')
    fnames = tf.data.Dataset.list_files(wav_glob)
    binaries = map(tf.io.read_file, fnames)
    const = tf.constant(label, dtype=tf.int8)
    for audio, _ in map(tf.audio.decode_wav, binaries):
        windows = tf.signal.frame(tf.reshape(audio, [-1]), frame_len, step)
        for frame in windows:
            yield _mel_frame(frame), const


def _accent_dataset(accent: str, label: int) -> tf.data.Dataset:
    """Process and generate dataset from all entries in accent."""
    return tf.data.Dataset.from_generator(
        partial(_frame_accent, accent, label),
        output_signature=(tf.TensorSpec(shape=(128, 59),
                                        dtype=tf.float32),
                          tf.TensorSpec(shape=(), dtype=tf.int8))
    )


def _fetch_accent(accent: str, label: int, hyp: Dict[str, Union[float, int]]) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Generate train and test split for accent."""
    num_files = len(listdir(path.join(DATA_DIR, accent)))
    test_sz = hyp['test_set_percent'] * num_files // 100
    dataset = _accent_dataset(accent, label)
    return dataset.skip(test_sz), dataset.take(test_sz)


def fetch_data(hyp: Dict[str, Union[float, int]]) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Fetch and split accents into datasets."""
    accents = sorted(listdir(DATA_DIR))
    train_set, test_set = _fetch_accent(accents[0], 0, hyp)
    for label, accent in enumerate(accents[1:], start=1):
        train, test = _fetch_accent(accent, label, hyp)
        train_set.concatenate(train)
        test_set.concatenate(test)
    return train_set, test_set
