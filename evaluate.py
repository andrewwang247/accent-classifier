"""
Evaluate performance of model.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, List
from matplotlib import pyplot as plt  # type: ignore


def plot_history(history: Dict[str, List[int]], dpi: int):
    """Plot loss and accuracy over training."""
    for metric in ('loss', 'sparse_categorical_accuracy'):
        short_name = metric.split('_')[-1]
        plt.figure(dpi=dpi)
        plt.plot(history[metric])
        plt.plot(history[f'val_{metric}'])
        plt.legend([short_name, f'val_{short_name}'])
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title(f'{short_name.capitalize()} over Training')
        plt.savefig(f'{short_name}.png')
