"""
Generate histogram of frame norms from dataset.

Copyright 2021. Siwei Wang.
"""
from itertools import chain
import numpy as np  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import tensorflow as tf  # type: ignore
from click import command, option  # type: ignore
from preprocess import load_accents
from train import get_hyp
# pylint: disable=no-value-for-parameter


@command()
@option('--n_bins', '-n', type=int, default=50,
        help='Number of histogram bins.')
def explore(n_bins: int):
    """Generate histogram of frame norms from dataset."""
    hyp = get_hyp()
    hyp['norm_threshold'] = 0
    train, val, test = load_accents(hyp)
    norms = np.fromiter((tf.norm(ds) for ds, _ in
                         chain(train, val, test)),
                        dtype=float)
    _, bins = np.histogram(norms, bins=n_bins)
    plt.figure(dpi=hyp['plot_dpi'])
    plt.hist(norms, bins=bins)
    plt.xlabel('Euclidean Norm')
    plt.ylabel('Frequency')
    plt.title('Histogram of Frame Norms')
    plt.savefig('norms.png')


if __name__ == '__main__':
    explore()
