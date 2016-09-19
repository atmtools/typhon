# -*- coding: utf-8 -*-

"""Testing the basic plotting functions.

This module provides basic functions to test plotting utilities.
"""

import os
import numpy as np
from typhon.plots import styles


class TestPlots(object):
    """Testing plotting functions."""

    def test_styles(self):
        """Test styles."""
        stylelist = ['typhon', 'typhon-geometry', 'typhon-fullcycler']
        assert np.any([os.path.isfile(styles(s)) for s in stylelist])
