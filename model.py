"""
Model definitions and getters.

Copyright 2021. Siwei Wang.
"""
from tensorflow.keras import Model, Sequential  # type: ignore
from tensorflow.keras.layers import Bidirectional, Dense, LSTM  # type: ignore
from tensorflow.keras.regularizers import l1_l2  # type: ignore


def _lstm_layer(units: int, ret_seq: bool) -> LSTM:
    """Create a regularized LSTM layer."""
    return LSTM(units, return_sequences=ret_seq,
                kernel_regularizer=l1_l2(0.02, 0.03),
                recurrent_regularizer=l1_l2(0.02, 0.03),
                bias_regularizer=l1_l2(0.02, 0.03),
                dropout=0.35, recurrent_dropout=0.35)


def get_bilstm(num_labels: int) -> Model:
    """Define and retrieve BiLSTM model."""
    return Sequential(
        layers=[Bidirectional(_lstm_layer(32, True)),
                Bidirectional(_lstm_layer(32, False)),
                Dense(num_labels, activation='softmax')],
        name='BiLSTM')
