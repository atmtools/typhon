from os.path import dirname, join

import numpy as np
from typhon.spareice import collocate, collocate_datasets, Dataset


class TestCollocator:
    """Testing the dataset methods."""

    datasets = None
    refdir = join(dirname(__file__), 'reference')
