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


def hyperparams() -> Dict[str, Union[float, int]]:
    """Load and return hyperparameters dictionary."""
    with open('hyperparameters.json') as fin:
        hyp = load(fin)
    return hyp


def data_shape(dataset: tf.data.Dataset) -> Tuple[int, ...]:
    """Get shape of data from dataset."""
    entry, _ = next(iter(dataset.as_numpy_iterator()))
    return entry.shape


def standardize(data: tf.Tensor) -> tf.Tensor:
    """Rescale data to have mean 0 and std 1."""
    mean = tf.reduce_mean(data)
    std = tf.math.reduce_std(data)
    denom = std if std != 0.0 else 1e-5
    return (data - mean) / denom


def compute_steps(hyp: Dict[str, Union[float, int]]) \
        -> Tuple[int, int, int]:
    """Compute number of steps per epoch for train, val, test."""
    batches = hyp['total_frames'] / hyp['batch_size']
    val = hyp['val_split']
    test = hyp['test_split']
    train = 1 - val - test
    train_steps, val_steps, test_steps = [
        round(spl * batches) for spl in (train, val, test)]
    return train_steps, val_steps, test_steps


def get_model(cnn: bool,
              in_shape: Tuple[int, ...],
              num_labels: int) -> Model:
    """Load model corresponding to given architecture."""
    return get_cnn_bilstm(in_shape[1:], num_labels) if cnn \
        else get_bilstm(num_labels)
