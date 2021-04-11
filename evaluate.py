"""
Evaluate performance of best model.

Copyright 2021. Siwei Wang.
"""
from tensorflow.keras.optimizers import Nadam  # type: ignore
from util import ACCENTS, data_shape, hyperparameters
from model import get_bilstm
from preprocess import load_accents


def eval_best_bilstm(save_path: str):
    """Evaluate the best bilstm on all data."""
    hyp = hyperparameters()
    train, val, test = load_accents(hyp)
    dataset = train.concatenate(val).concatenate(test) \
        .batch(hyp['batch_size'], drop_remainder=True) \
        .prefetch(1)
    model = get_bilstm(len(ACCENTS))
    model.build(data_shape(dataset))
    model.load_weights(save_path)
    model.compile(optimizer=Nadam(hyp['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.evaluate(dataset)


if __name__ == '__main__':
    eval_best_bilstm('bilstm.hdf5')
