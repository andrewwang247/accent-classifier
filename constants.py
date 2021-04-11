"""
List of constant variables.

Copyright 2021. Siwei Wang.
"""
from typing import Dict, Union
from json import load
from os import scandir

DATA_DIR = 'recordings'
ACCENTS = sorted(acc.name for acc in scandir(DATA_DIR))
# ACCENTS.remove('english')


def hyperparameters() -> Dict[str, Union[float, int]]:
    """Load and return hyperparameters dictionary."""
    with open('hyperparameters.json') as fin:
        hyp = load(fin)
    return hyp
