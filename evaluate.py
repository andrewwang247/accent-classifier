"""
Evaluate performance of model.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, List
from matplotlib import pyplot as plt  # type: ignore


def plot_history(history: Dict[str, List[int]], dpi: int):
    """Plot loss and accuracy over training."""
    for metric in ('loss', 'accuracy'):
        plt.figure(dpi=dpi)
        plt.plot(history[metric])
        plt.plot(history[f'val_{metric}'])
        plt.legend([metric, f'val_{metric}'])
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric.capitalize()} Value')
        plt.title(f'{metric.capitalize()} over Training')
        plt.savefig(f'{metric}.png')
