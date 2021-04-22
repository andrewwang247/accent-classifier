"""
Model definitions and getters.

Copyright 2021. Siwei Wang.
"""
from typing import List, Tuple
import tensorflow as tf  # type: ignore
from tensorflow.keras import Model, Sequential, layers  # type: ignore
from tensorflow.keras.regularizers import l2, Regularizer  # type: ignore
from tensorflow.keras.initializers import LecunNormal  # type: ignore


def _regularizer() -> Regularizer:
    """Create a regularizer."""
    return l2(5e-2)


def _conv_layer(filters: int, kernel_sz: int) -> layers.Conv2D:
    """Create a regularized Conv2D layer."""
    return layers.Conv2D(filters, kernel_sz, padding='same',
                         activation='selu',
                         kernel_initializer=LecunNormal(),
                         kernel_regularizer=_regularizer())


def _lstm_layer(units: int, ret_seq: bool) -> layers.LSTM:
    """Create a regularized LSTM layer."""
    return layers.LSTM(units, return_sequences=ret_seq,
                       kernel_regularizer=_regularizer(),
                       recurrent_regularizer=_regularizer(),
                       dropout=0.2, recurrent_dropout=0.4)


def _global_depth_pool(pool_op: str) -> layers.Lambda:
    """Global (avg or max) depth pooling and reshape."""
    assert pool_op in ('avg', 'max')
    func = tf.reduce_mean if pool_op == 'avg' else tf.reduce_max
    return layers.Lambda(lambda tensor: func(tensor, axis=-1),
                         name='global_depth_pool')


def _cnn_layers(in_shape: Tuple[int, ...]) \
        -> List[layers.Layer]:
    """Get a list of layers for CNN without final predictor."""
    num_max_pools = 3  # adjust this number as needed.
    out_shape = (in_shape[0] // 2**num_max_pools, -1)
    cnn_lays = [layers.Reshape((*in_shape, 1),
                               input_shape=in_shape,
                               name='create_channel'),
                _conv_layer(16, 5),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Dropout(0.3),
                _conv_layer(32, 4),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Dropout(0.3),
                _conv_layer(32, 3),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Dropout(0.3),
                _conv_layer(16, 3),
                layers.Dropout(0.3),
                layers.Reshape(out_shape,
                               name='sequence_flatten')]
    assert sum(isinstance(lay, layers.MaxPool2D)
               for lay in cnn_lays) == num_max_pools
    return cnn_lays


def _lstm_layers(num_labels: int) -> List[layers.Layer]:
    """Get a list of layers for LSTM with final predictor."""
    return [layers.Bidirectional(_lstm_layer(24, True)),
            layers.Bidirectional(_lstm_layer(18, True)),
            layers.Bidirectional(_lstm_layer(12, False)),
            layers.Dense(num_labels, activation='softmax')]


def get_bilstm(num_labels: int) -> Model:
    """Define and retrieve BiLSTM model."""
    return Sequential(
        layers=_lstm_layers(num_labels),
        name='bilstm')


def get_cnn_bilstm(in_shape: Tuple[int, ...],
                   num_labels: int) -> Model:
    """Define and retrive CNN_BiLSTM model."""
    return Sequential(
        layers=_cnn_layers(in_shape) + _lstm_layers(num_labels),
        name='cnn_bilstm')
