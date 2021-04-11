"""
Train models on data.

Copyright 2021. Siwei Wang.
"""
from tensorflow.keras import optimizers, callbacks  # type: ignore
from preprocess import ACCENTS, load_accents
from evaluate import plot_history
from model import get_bilstm
from constants import hyperparameters
# pylint: disable=redefined-outer-name


def train():
    """Train models on data."""
    hyp = hyperparameters()
    train, val, test = [ds.repeat().shuffle(hyp['shuffle_buffer'])
                        .batch(hyp['batch_size'], drop_remainder=True)
                        .prefetch(1)
                        for ds in load_accents(hyp)]

    model = get_bilstm(len(ACCENTS))
    model.compile(optimizer=optimizers.Nadam(hyp['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = callbacks.ModelCheckpoint('bilstm.hdf5',
                                           monitor='val_loss',
                                           save_best_only=True)
    hist = model.fit(train, epochs=hyp['epochs'], callbacks=[checkpoint],
                     steps_per_epoch=hyp['train_steps'],
                     validation_data=val, validation_steps=hyp['val_steps'],
                     workers=hyp['cpu_cores'], use_multiprocessing=True)
    model.evaluate(test, steps=hyp['test_steps'])
    model.summary()

    dpi = hyp['plot_dpi']
    assert hist is not None
    assert isinstance(dpi, int)
    plot_history(hist.history, dpi)


if __name__ == '__main__':
    train()
