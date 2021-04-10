"""
Bidirectional LSTM model.

Copyright 2021. Siwei Wang.
"""
from typing import Tuple
from tensorflow.keras import layers, regularizers  # type: ignore
from tensorflow.keras import Model, Sequential


def _get_bilstm_layer(units: int, ret_seq: bool) -> layers.Layer:
    """Create l1_l2 regularized BiLSTM layer."""
    return layers.Bidirectional(
        layers.LSTM(units, return_sequences=ret_seq,
                    kernel_regularizer=regularizers.l1_l2(),
                    recurrent_regularizer=regularizers.l1_l2())
    )


def get_bilstm(in_shape: Tuple[int, ...], num_labels: int) -> Model:
    """Define and retrieve BiLSTM model."""
    model = Sequential(
        layers=[layers.Reshape((-1, 1), input_shape=in_shape),
                _get_bilstm_layer(10, True),
                layers.Dropout(0.25),
                _get_bilstm_layer(10, False),
                layers.Dropout(0.25),
                layers.Dense(num_labels, activation='softmax')],
        name='BiLSTM')
    return model
