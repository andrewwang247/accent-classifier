"""
Train models on data.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, Union
from json import load
from tensorflow import keras  # type: ignore
from preprocess import ACCENTS, load_accents, MEL_SHAPE
from bilstm import get_bilstm
# pylint: disable=redefined-outer-name

HYP_FILE = 'hyperparameters.json'


def train():
    """Train models on data."""
    with open(HYP_FILE) as fp_hyp:
        hyp: Dict[str, Union[float, int]] = load(fp_hyp)
    batch_size = hyp['batch_size']
    train, test = [ds.repeat().batch(batch_size)
                   for ds in load_accents(hyp)]

    model = get_bilstm(MEL_SHAPE, len(ACCENTS))
    model.build((batch_size, *MEL_SHAPE))
    model.compile(optimizer=keras.optimizers.Adam(hyp['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint('saved/bilstm_{epoch:02d}.h5')
    model.fit(train, batch_size=batch_size,
              epochs=hyp['epochs'], callbacks=[checkpoint],
              steps_per_epoch=hyp['steps_per_epoch'],
              workers=12, use_multiprocessing=True)
    model.evaluate(test, batch_size=batch_size)


if __name__ == '__main__':
    train()
