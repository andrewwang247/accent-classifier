"""
Constant variables and utility functions.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, Tuple, Union
from json import load
from os import scandir
import tensorflow as tf  # type: ignore
from tensorflow.keras import Model  # type: ignore
from model import get_bilstm, get_cnn, get_cnn_bilstm

DATA_DIR = 'recordings'
ACCENTS = sorted(acc.name for acc in scandir(DATA_DIR))
# ACCENTS.remove('english')


def hyperparameters() -> Dict[str, Union[float, int]]:
    """Load and return hyperparameters dictionary."""
    with open('hyperparameters.json') as fin:
        hyp = load(fin)
    return hyp


def data_shape(dataset: tf.data.Dataset) -> Tuple[int, ...]:
    """Get shape of data from dataset."""
    entry, _ = next(iter(dataset.as_numpy_iterator()))
    return entry.shape


def get_model(architecture: str,
              in_shape: Tuple[int, ...],
              num_labels: int) -> Model:
    """Load model corresponding to given architecture."""
    assert architecture in ('bilstm', 'cnn', 'cnn_bilstm')
    if architecture == 'bilstm':
        return get_bilstm(num_labels)
    if architecture == 'cnn':
        return get_cnn(in_shape[1:], num_labels)
    return get_cnn_bilstm(in_shape[1:], num_labels)
