"""
Visualize mel spectrogram input data.

Copyright 2021. Siwei Wang.
"""
from os import path
from matplotlib import pyplot as plt  # type: ignore
from seaborn import heatmap  # type: ignore
from click import command, option  # type: ignore
import tensorflow as tf  # type: ignore
from util import ARTIFACT_DIR, hyperparams
from preprocess import load_accents
# pylint: disable=no-value-for-parameter


@command()
@option('--num', '-n', type=int, required=True,
        help='The number of plots to create.')
def main(num: int):
    """Visualize mel spectrogram input data."""
    hyp = hyperparams()
    train, val, test = load_accents(hyp)
    dataset = train.concatenate(val) \
        .concatenate(test) \
        .shuffle(hyp['shuffle_buffer']) \
        .take(num).prefetch(1)
    plt.rcParams['figure.dpi'] = hyp['plot_dpi']
    for idx, (spec, _) in enumerate(dataset):
        fpath = path.join(ARTIFACT_DIR, f'spec_{idx:02d}.png')
        plt.figure()
        heatmap(tf.transpose(spec))
        plt.savefig(fpath)
        plt.close()


if __name__ == '__main__':
    main()
