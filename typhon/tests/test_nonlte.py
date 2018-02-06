# -*- coding: utf-8 -*-
"""Testing the basic nonlte functions.
"""
import numpy as np

from typhon import nonlte


class TestNonlte:
    """Testing the nonlte functions."""
    def test_trapz_inte_edge(self):
        """Check trapzouidal integration function."""
        x = np.arange(1, 11, 1)
        y = np.random.random(10)

        area = nonlte.mathmatics.trapz_inte_edge(y, x)
        area_ref = np.trapz(y, x)
        assert np.allclose(area.sum(), area_ref)
