"""
Count and set number of frames.

Copyright 2021. Siwei Wang.
"""
from json import dump
from util import hyperparams
from preprocess import load_accents


def main():
    """Count and set number of frames."""
    hyp = hyperparams()
    train, val, test = load_accents(hyp)
    data = train.concatenate(val).concatenate(test)
    frames = sum(1 for _ in data)
    print(f'Counted {frames} frames.')
    hyp['total_frames'] = frames
    with open('hyperparameters.json', 'w') as fin:
        dump(hyp, fin, indent=2)


if __name__ == '__main__':
    main()
