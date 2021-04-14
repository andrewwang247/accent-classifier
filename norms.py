"""
Plot histogram of L2 norms for audio.

Copyright 2021. Siwei Wang.
"""
from os import path
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from click import command, option
from preprocess import _decode_wav_files, _frame_audio
from util import DATA_DIR, hyperparameters
# pylint: disable=no-value-for-parameter


@command()
@option('--bins', '-b', type=int, required=True,
        help='Number of bins in histogram.')
def plot_norms(bins: int):
    """Plot histogram of L2 norms for audio."""
    hyp = hyperparameters()
    acc_glob = path.join(DATA_DIR, '*', '*.wav')
    file_list = tf.data.Dataset.list_files(acc_glob)
    audio = _decode_wav_files(file_list)
    frames = _frame_audio(audio, hyp)
    norms = frames.map(tf.norm)
    data = np.fromiter((norm.numpy() for norm in norms),
                       dtype=float)
    plot_dpi = hyp['plot_dpi']
    assert isinstance(plot_dpi, int)
    plt.figure(dpi=plot_dpi)
    plt.hist(data, bins)
    plt.xlabel('Euclidean Norm')
    plt.ylabel('Frequency')
    plt.title('Histogram of Norms from Audio Frames')
    plt.savefig('norms.png')
    plt.close()


if __name__ == '__main__':
    plot_norms()
