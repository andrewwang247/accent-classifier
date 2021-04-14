"""
Train model on data. Evaluate test set on best models.

Copyright 2021. Siwei Wang.
"""
from os import path
from typing import Dict, List
from json import dump
from matplotlib import pyplot as plt  # type: ignore
from tensorflow.keras import optimizers, callbacks  # type: ignore
from click import command, option  # type: ignore
from preprocess import dataset_class_weights, load_accents
from util import ACCENTS, ARTIFACT_DIR
from util import hyperparameters, data_shape, get_model
# pylint: disable=redefined-outer-name,no-value-for-parameter


def plot_history(history: Dict[str, List[int]],
                 metrics: List[str], model_name: str, dpi: int):
    """Plot training and validation metrics and loss."""
    for metric in metrics + ['loss']:
        plt.figure(dpi=dpi)
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
    hyp = hyperparameters()
    train, val, test = [ds.repeat().shuffle(hyp['shuffle_buffer'])
                        .batch(hyp['batch_size'], drop_remainder=True)
                        .prefetch(1)
                        for ds in load_accents(hyp)]
    in_shape = data_shape(train)

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
    weights = dataset_class_weights(ACCENTS)
    try:
        hist = model.fit(
            train,
            epochs=hyp['epochs'],
            class_weight=weights,
            steps_per_epoch=hyp['train_steps'],
            callbacks=checkpoints,
            validation_data=val,
            validation_steps=hyp['val_steps'],
            workers=hyp['cpu_cores'],
            use_multiprocessing=True)

        assert hist is not None
        hist_path = path.join(ARTIFACT_DIR, f'{model.name}_history.json')
        with open(hist_path, 'w') as fp_out:
            dump(hist.history, fp_out, indent=2)
        dpi = hyp['plot_dpi']
        assert isinstance(dpi, int)
        plot_history(hist.history, tracked_metrics, model.name, dpi)
    finally:
        for met in tracked_metrics + ['loss']:
            print(f'Evaluating {model.name} with best {met}...')
            model.load_weights(
                path.join(ARTIFACT_DIR, f'{model.name}_{met}.hdf5'))
            model.evaluate(test, steps=hyp['test_steps'])


if __name__ == '__main__':
    train()
