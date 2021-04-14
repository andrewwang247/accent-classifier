"""
Make predictions using best saved model.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, Union
from os import path
from pprint import pprint
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import tensorflow_io as tfio  # type: ignore
from click import command, option, Path  # type: ignore
from util import ARTIFACT_DIR, ACCENTS
from util import get_model, hyperparameters
# pylint: disable=no-value-for-parameter


def load_from_path(fpath: str,
                   hyp: Dict[str, Union[float, int]]) \
        -> tf.Tensor:
    """Preprocess a single audio file for prediction."""
    contents = tf.io.read_file(fpath)
    audio, _ = tf.audio.decode_wav(contents)
    frames = tf.signal.frame(tf.reshape(audio, [-1]),
                             hyp['frame_len'],
                             hyp['frame_step'])
    specs = [tfio.experimental.audio
             .spectrogram(frame, hyp['num_fft'],
                          hyp['spec_window'],
                          hyp['spec_stride'])
             for frame in frames]
    mels = [tfio.experimental.audio
            .melscale(spec, hyp['sampling_rate'],
                      hyp['num_mels'], hyp['freq_min'],
                      hyp['freq_max'])
            for spec in specs]
    return tf.convert_to_tensor(mels, dtype=tf.float32)


def weight_path(cnn: bool, accuracy: bool) -> str:
    """Get path to model weights."""
    prefix = 'cnn_' if cnn else ''
    core = 'bilstm_'
    suffix = 'accuracy' if accuracy else 'loss'
    fname = prefix + core + suffix + '.hdf5'
    return path.join(ARTIFACT_DIR, fname)


@command()
@option('--cnn', '-c', is_flag=True,
        help='Flag to use CNN-BiLSTM model.')
@option('--fpath', '-f', required=True,
        type=Path(exists=True, file_okay=True,
                  dir_okay=True, readable=True),
        help='Path to file to make prediction on.')
@option('--accuracy/--loss', '-a/-l', default=False,
        help='Choose between model with best accuracy or loss')
def predict(cnn: bool, fpath: str, accuracy: bool):
    """Make predictions using best saved model."""
    hyp = hyperparameters()
    in_data = load_from_path(fpath, hyp)
    model = get_model(cnn, in_data.shape, len(ACCENTS))
    model.build(in_data.shape)
    model.load_weights(weight_path(cnn, accuracy))
    preds = model.predict(in_data,
                          workers=hyp['cpu_cores'],
                          use_multiprocessing=True)
    combined = np.sum(preds, axis=0)
    scores = dict(zip(ACCENTS, combined / np.sum(combined)))
    winner = np.argmax(combined)
    print('Prediction:', ACCENTS[winner])
    pprint(scores)


if __name__ == '__main__':
    predict()
