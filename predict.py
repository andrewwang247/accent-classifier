"""
Make soft ensemble predictions using best model.

Copyright 2021. Siwei Wang.
"""
from os import path
from typing import Dict, Union
import tensorflow as tf  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from click import command, option, Path
from preprocess import _transform_files
from util import ACCENTS, ARTIFACT_DIR, hyperparams, get_model
# pylint: disable=no-value-for-parameter


def preproc_file(audio: str,
                 hyp: Dict[str, Union[float, int]]) \
        -> tf.Tensor:
    """Preprocess audio file into array."""
    in_file = tf.data.Dataset.list_files(audio)
    lbl_specs = _transform_files(in_file, 0, hyp)
    return tf.convert_to_tensor([tup[0] for tup in lbl_specs],
                                dtype=tf.float32)


def load_best_model(cnn: bool, in_shape: tf.TensorShape) \
        -> tf.keras.Model:
    """Load best BiLSTM or CNN-BiLSTM model."""
    model = get_model(cnn, in_shape, len(ACCENTS))
    model.build(in_shape)
    weight_file = path.join(ARTIFACT_DIR, f'{model.name}_loss.hdf5')
    model.load_weights(weight_file)
    return model


def plot_scores(aggregate: tf.Tensor, model_name: str):
    """Make a plot of aggregate label probabilities."""
    scores = aggregate / tf.reduce_sum(aggregate)
    plt.figure()
    plt.bar([acc[:3] for acc in ACCENTS], scores)
    plt.xlabel('Accent')
    plt.ylabel('Score')
    plt.title(f'Prediction Scores by {model_name}')
    plt.savefig(path.join(ARTIFACT_DIR, 'prediction.png'))
    plt.close()


@command()
@option('--audio', '-a', type=Path(exists=True,
                                   file_okay=True,
                                   dir_okay=False,
                                   readable=True),
        required=True, help='Path to audio recording.')
@option('--cnn', '-c', is_flag=True,
        help='Set flag to predict with CNN-BiLSTM.')
def predict(audio: str, cnn: bool):
    """Make soft ensemble predictions using best model."""
    hyp = hyperparams()
    plt.rcParams['figure.dpi'] = hyp['plot_dpi']
    specs = preproc_file(audio, hyp)
    model = load_best_model(cnn, specs.shape)
    scores = model.predict(specs)
    aggregate = tf.reduce_sum(scores, axis=0)
    prediction = ACCENTS[tf.argmax(aggregate)]
    print('Predicted accent:', prediction)
    plot_scores(aggregate, model.name)


if __name__ == '__main__':
    predict()
