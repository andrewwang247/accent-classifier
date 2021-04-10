"""
Bidirectional LSTM model.

Copyright 2021. Siwei Wang.
"""
from typing import List, Tuple
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras import Model, Sequential


def _get_bilstm_layers(in_shape: Tuple[int, ...],
                       num_labels: int) -> List[layers.Layer]:
    return [
        layers.Reshape((-1, 1), input_shape=in_shape),
        layers.Bidirectional(
            layers.LSTM(20, return_sequences=True)),
        layers.Dropout(0.25),
        layers.Bidirectional(layers.LSTM(20)),
        layers.Dropout(0.25),
        layers.Dense(num_labels, activation='softmax')]


def get_bilstm(in_shape: Tuple[int, ...], num_labels: int) -> Model:
    """Retrieve BiLSTM model."""
    model = Sequential(
        layers=_get_bilstm_layers(in_shape, num_labels),
        name='BiLSTM')
    return model
