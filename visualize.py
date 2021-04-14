"""
Visualize mel spectrogram input data.

Copyright 2021. Siwei Wang.
"""
from os import remove, listdir, path
from matplotlib import pyplot as plt  # type: ignore
from seaborn import heatmap  # type: ignore
from click import command, option  # type: ignore
from util import hyperparameters
from preprocess import load_accents
# pylint: disable=no-value-for-parameter

VIZ_DIR = 'spectrograms'


@command()
@option('--num', '-n', type=int, required=True,
        help='The number of plots to create.')
def main(num: int):
    """Visualize mel spectrogram input data."""
    for mel in listdir(VIZ_DIR):
        remove(path.join(VIZ_DIR, mel))
    hyp = hyperparameters()
    train, val, test = load_accents(hyp)
    dataset = train.concatenate(val) \
        .concatenate(test) \
        .shuffle(hyp['shuffle_buffer']) \
        .take(num).prefetch(1)
    for idx, (audio, lbl) in enumerate(dataset):
        fpath = path.join(VIZ_DIR, f'{idx:02d}_{lbl.numpy()[0]}.png')
        plt.figure(dpi=hyp['plot_dpi'])
        heatmap(audio)
        plt.savefig(fpath)
        plt.close()


if __name__ == '__main__':
    main()
