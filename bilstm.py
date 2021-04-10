"""
Bidirectional LSTM model.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, Union
from tensorflow.keras import layers, regularizers  # type: ignore
from tensorflow.keras import Model, Sequential


def _get_bilstm_layer(units: int, ret_seq: bool,
                      hyp: Dict[str, Union[float, int]]) -> layers.Layer:
    """Create l1_l2 regularized BiLSTM layer."""
    reg_l1 = hyp['reg_l1']
    reg_l2 = hyp['reg_l2']
    return layers.Bidirectional(
        layers.LSTM(units, return_sequences=ret_seq,
                    kernel_regularizer=regularizers.l1_l2(reg_l1, reg_l2),
                    recurrent_regularizer=regularizers.l1_l2(reg_l1, reg_l2))
    )


def get_bilstm(num_labels: int, hyp: Dict[str, Union[float, int]]) -> Model:
    """Define and retrieve BiLSTM model."""
    model = Sequential(
        layers=[_get_bilstm_layer(35, True, hyp),
                layers.Dropout(0.2),
                _get_bilstm_layer(35, True, hyp),
                layers.Dropout(0.2),
                _get_bilstm_layer(35, False, hyp),
                layers.Dropout(0.2),
                layers.Dense(num_labels, activation='softmax')],
        name='BiLSTM')
    return model
