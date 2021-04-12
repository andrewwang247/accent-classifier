"""
Train models on data.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, List
from pickle import dump
from matplotlib import pyplot as plt  # type: ignore
from tensorflow.keras import optimizers, callbacks  # type: ignore
from preprocess import dataset_class_weights, load_accents
from model import get_bilstm
from util import ACCENTS, hyperparameters, data_shape
# pylint: disable=redefined-outer-name


def plot_history(history: Dict[str, List[int]],
                 metrics: List[str], dpi: int):
    """Plot loss and accuracy over training."""
    for metric in metrics + ['loss']:
        plt.figure(dpi=dpi)
        plt.plot(history[metric])
        plt.plot(history[f'val_{metric}'])
        plt.legend([metric, f'val_{metric}'])
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric.capitalize()} Value')
        plt.title(f'{metric.capitalize()} over Training')
        plt.savefig(f'{metric}.png')


def train():
    """Train model on data. Evaluate test set on best model."""
    hyp = hyperparameters()
    train, val, test = [ds.repeat().shuffle(hyp['shuffle_buffer'])
                        .batch(hyp['batch_size'], drop_remainder=True)
                        .prefetch(1)
                        for ds in load_accents(hyp)]

    tracked_metrics = ['accuracy']
    model = get_bilstm(len(ACCENTS))
    model.build(data_shape(test))
    model.compile(optimizer=optimizers.Nadam(hyp['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=tracked_metrics)
    model.summary()

    checkpoints = [callbacks.ModelCheckpoint(f'bilstm_{met}.hdf5',
                                             monitor=f'val_{met}',
                                             save_best_only=True)
                   for met in tracked_metrics + ['loss']]
    weights = dataset_class_weights(ACCENTS)
    hist = model.fit(train, epochs=hyp['epochs'], class_weight=weights,
                     steps_per_epoch=hyp['train_steps'], callbacks=checkpoints,
                     validation_data=val, validation_steps=hyp['val_steps'],
                     workers=hyp['cpu_cores'], use_multiprocessing=True)

    assert hist is not None
    with open('history.pickle', 'wb') as pick:
        dump(hist.history, pick)
    dpi = hyp['plot_dpi']
    assert isinstance(dpi, int)
    plot_history(hist.history, tracked_metrics, dpi)
    for met in tracked_metrics + ['loss']:
        print(f'Evaluating BiLSTM with best {met}...')
        model.load_weights(f'bilstm_{met}.hdf5')
        model.evaluate(test, steps=hyp['test_steps'])


if __name__ == '__main__':
    train()
