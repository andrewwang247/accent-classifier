"""
Train model on data. Evaluate test set on best models.

Copyright 2021. Siwei Wang.
"""
from os import path, scandir
from typing import Dict, List
from json import dump
import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from tensorflow.keras import optimizers, callbacks  # type: ignore
from click import command, option  # type: ignore
from preprocess import load_accents
from util import ACCENTS, ARTIFACT_DIR, DATA_DIR
from util import hyperparams, data_shape, get_model, compute_steps
# pylint: disable=redefined-outer-name,no-value-for-parameter


def compute_class_weights(accents: List[str]) -> Dict[int, float]:
    """Compute class weights for accents."""
    counts = np.empty(len(accents), dtype=int)
    for idx, accent in enumerate(accents):
        wav_path = path.join(DATA_DIR, accent)
        counts[idx] = sum(1 for _ in scandir(wav_path))
    weights = np.sum(counts) / counts / len(accents)
    return dict(enumerate(weights / np.sum(weights)))


def dump_history(hist: Dict[str, List[float]],
                 model_name: str):
    """Save history from training."""
    hist_path = path.join(ARTIFACT_DIR, f'{model_name}_history.json')
    with open(hist_path, 'w') as fp_out:
        dump(hist, fp_out, indent=2)


def plot_history(history: Dict[str, List[int]],
                 metrics: List[str], model_name: str):
    """Plot training and validation metrics and loss."""
    for metric in metrics + ['loss']:
        plt.figure()
        plt.plot(history[metric])
        plt.plot(history[f'val_{metric}'])
        plt.legend([metric, f'val_{metric}'])
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric.capitalize()} Value')
        plt.title(f'{metric.capitalize()} over Training')
        plt.savefig(path.join(ARTIFACT_DIR,
                              f'{model_name}_{metric}.png'))
        plt.close()


@command()
@option('--cnn', '-c', is_flag=True,
        help='Set flag to train CNN-BiLSTM.')
def train(cnn: bool):
    """Train model on data. Evaluate test set on best models."""
    hyp = hyperparams()
    plt.rcParams['figure.dpi'] = hyp['plot_dpi']
    train, val, test = [ds.repeat().shuffle(hyp['shuffle_buffer'])
                        .batch(hyp['batch_size'], drop_remainder=True)
                        .prefetch(1)
                        for ds in load_accents(hyp)]
    in_shape = data_shape(val)

    tracked_metrics = ['accuracy']
    model = get_model(cnn, in_shape, len(ACCENTS))
    model.build(in_shape)
    model.compile(optimizer=optimizers.Nadam(hyp['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=tracked_metrics)
    model.summary()

    checkpoints = [callbacks.ModelCheckpoint(
        path.join(ARTIFACT_DIR, f'{model.name}_{met}.hdf5'),
        monitor=f'val_{met}', save_best_only=True)
        for met in tracked_metrics + ['loss']]
    weights = compute_class_weights(ACCENTS)
    train_steps, val_steps, test_steps = compute_steps(hyp)
    try:
        hist = model.fit(
            train,
            epochs=hyp['epochs'],
            class_weight=weights,
            steps_per_epoch=train_steps,
            callbacks=checkpoints,
            validation_data=val,
            validation_steps=val_steps,
            workers=hyp['cpu_cores'],
            use_multiprocessing=True)

        assert hist is not None
        dump_history(hist.history, model.name)
        plot_history(hist.history, tracked_metrics, model.name)
    finally:
        for met in tracked_metrics + ['loss']:
            print(f'Evaluating {model.name} with best {met}...')
            model.load_weights(
                path.join(ARTIFACT_DIR, f'{model.name}_{met}.hdf5'))
            model.evaluate(test, steps=test_steps)


if __name__ == '__main__':
    train()
