"""
Bidirectional LSTM model.

Copyright 2021. Siwei Wang.
"""
from tensorflow import keras  # type: ignore


def _get_bilstm_layer(units: int, ret_seq: bool) -> keras.layers.Layer:
    """Create l1_l2 regularized BiLSTM layer."""
    elastic_reg = keras.regularizers.l1_l2(0.01, 0.01)
    return keras.layers.Bidirectional(
        keras.layers.LSTM(units, return_sequences=ret_seq,
                          kernel_regularizer=elastic_reg,
                          recurrent_regularizer=elastic_reg,
                          recurrent_dropout=0.2))


def get_bilstm(num_labels: int) -> keras.Model:
    """Define and retrieve BiLSTM model."""
    return keras.Sequential(
        layers=[_get_bilstm_layer(128, True),
                keras.layers.Dropout(0.2),
                _get_bilstm_layer(128, True),
                keras.layers.Dropout(0.2),
                _get_bilstm_layer(128, False),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(num_labels, activation='softmax')],
        name='BiLSTM')
