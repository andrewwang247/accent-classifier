"""
Train models on data.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, List, Union
from json import load
from tensorflow import keras  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from preprocess import ACCENTS, load_accents
from bilstm import get_bilstm
# pylint: disable=redefined-outer-name

HYP_FILE = 'hyperparameters.json'
# Total number of signals in entire dataset
APPROX_TOTAL_SIGS = 240_000_000


def epoch_steps(source: str, hyp: Dict[str, Union[float, int]]) -> int:
    """Approximate epoch steps to cover the dataset."""
    assert source in ('train', 'val', 'test')
    frac = 1 - hyp['val_split'] - hyp['test_split'] \
        if source == 'train' else hyp[f'{source}_split']
    windows = frac * APPROX_TOTAL_SIGS / hyp['frame_step']
    return round(windows / hyp['batch_size'])


def plot_history(history: Dict[str, List[float]], fpath: str):
    """Save a matplotlib plot of training history."""
    plt.figure(dpi=1200)
    for points in history.values():
        plt.plot(points)
    plt.legend(history.keys())
    plt.xlabel('Epoch')
    plt.title('Training Plot')
    plt.savefig(fpath)


def train():
    """Train models on data."""
    with open(HYP_FILE) as fin:
        hyp: Dict[str, Union[float, int]] = load(fin)
    batch_size = hyp['batch_size']
    train, val, test = [ds.repeat()
                        .batch(batch_size, drop_remainder=True)
                        .prefetch(1)
                        for ds in load_accents(hyp)]

    in_shape = (hyp['frame_len'],)
    model = get_bilstm(in_shape, len(ACCENTS))
    model.build((batch_size, *in_shape))
    model.compile(optimizer=keras.optimizers.Adam(hyp['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint('saved/bilstm_{epoch:02d}.h5')
    hist = model.fit(train, batch_size=batch_size,
                     steps_per_epoch=epoch_steps('train', hyp),
                     epochs=hyp['epochs'], callbacks=[checkpoint],
                     validation_data=val, validation_batch_size=batch_size,
                     validation_steps=epoch_steps('val', hyp),
                     workers=12, use_multiprocessing=True)
    result = model.evaluate(test, batch_size=batch_size,
                            steps=epoch_steps('test', hyp))
    plot_history(hist.history, 'train_plot.png')
    print(f'Result: {result}')


if __name__ == '__main__':
    train()
