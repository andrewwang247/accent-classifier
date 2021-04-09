"""
Make predictions using the model.

Copyright 2021. Siwei Wang.
"""
from click import command, option, Path
# pylint: disable=no-value-for-parameter


@command()
@option('--fpath', '-f', type=Path(exists=True,
                                   file_okay=True,
                                   dir_okay=False,
                                   readable=True),
        required=True, help='A wav file to analyze.')
def predict(fpath: str):
    """Make predictions using the model."""
    print(fpath)


if __name__ == '__main__':
    predict()
