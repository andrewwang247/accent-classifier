"""
Constant variables and utility functions.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, Tuple, Union
from json import load
from os import scandir
import tensorflow as tf  # type: ignore
from tensorflow.keras import Model  # type: ignore
from model import get_bilstm, get_cnn_bilstm

DATA_DIR = 'recordings'
ACCENTS = sorted(acc.name for acc in scandir(DATA_DIR))
# ACCENTS.remove('english')
ARTIFACT_DIR = 'bin'


def hyperparameters() -> Dict[str, Union[float, int]]:
    """Load and return hyperparameters dictionary."""
    with open('hyperparameters.json') as fin:
        hyp = load(fin)
    return hyp


def data_shape(dataset: tf.data.Dataset) -> Tuple[int, ...]:
    """Get shape of data from dataset."""
    entry, _ = next(iter(dataset.as_numpy_iterator()))
    return entry.shape


def _standardize_tensor(tens: tf.Tensor) -> tf.Tensor:
    """Rescale tensor to 0 mean and 1 std."""
    mean = tf.reduce_mean(tens)
    std = tf.math.reduce_std(tens)
    denom = std if std != 0.0 else tf.constant(1e-4)
    return (tens - mean) / denom


def _normalize_tensor(tens: tf.Tensor) -> tf.Tensor:
    """Rescales int16 tensor to [-1, 1]."""
    as_float = tf.cast(tens, tf.float32)
    assert isinstance(as_float, tf.Tensor)
    max_int_16 = 2.0 ** 15
    return as_float / max_int_16


def get_model(cnn: bool,
              in_shape: Union[Tuple[int, ...], tf.TensorShape],
              num_labels: int) -> Model:
    """Load model corresponding to given architecture."""
    return get_cnn_bilstm(in_shape[1:], num_labels) if cnn \
        else get_bilstm(num_labels)
