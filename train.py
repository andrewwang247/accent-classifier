"""
Train models on data.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, Tuple, Union
from json import load
import tensorflow as tf  # type: ignore
from tensorflow.keras import optimizers, callbacks  # type: ignore
from preprocess import ACCENTS, load_accents
from evaluate import plot_history
from bilstm import get_bilstm
# pylint: disable=redefined-outer-name

HYP_FILE = 'hyperparameters.json'
# Total number of signals in entire dataset
APPROX_TOTAL_SIGS = 240_000_000


def get_hyp() -> Dict[str, Union[float, int]]:
    """Retrieve hyperparameter dictionary."""
    with open(HYP_FILE) as fin:
        hyp = load(fin)
    return hyp


def input_shape(dataset: tf.data.Dataset) -> Tuple[int, ...]:
    """Get the input shape of the data entries."""
    in_batch, _ = next(iter(dataset.as_numpy_iterator()))
    return in_batch.shape


def train():
    """Train models on data."""
    hyp = get_hyp()
    batch_size = hyp['batch_size']
    train, val, test = [ds.shuffle(hyp['shuffle_buffer']).repeat()
                        .batch(batch_size, drop_remainder=True)
                        .prefetch(1)
                        for ds in load_accents(hyp)]

    # This shape depends on hyperparameters. Adjust if necessary!
    model = get_bilstm(len(ACCENTS), hyp)
    model.build(input_shape(val))
    model.compile(optimizer=optimizers.Nadam(hyp['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    model.summary()

    checkpoint = callbacks.ModelCheckpoint('saved/bilstm_{epoch:02d}.h5')
    hist = model.fit(train, batch_size=batch_size,
                     steps_per_epoch=hyp['train_steps'],
                     epochs=hyp['epochs'], callbacks=[checkpoint],
                     validation_data=val, validation_batch_size=batch_size,
                     validation_steps=hyp['val_steps'],
                     workers=hyp['cpu_cores'], use_multiprocessing=True)
    test_loss, test_acc = model.evaluate(test, batch_size=batch_size,
                                         steps=hyp['test_steps'])
    dpi = hyp['plot_dpi']
    assert hist is not None
    assert isinstance(dpi, int)
    plot_history(hist.history, dpi)
    print(f'Test results: loss={test_loss}, acc={test_acc}')


if __name__ == '__main__':
    train()
