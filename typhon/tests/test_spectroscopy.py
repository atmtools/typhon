# -*- coding: utf-8 -*-
"""Testing the functions in typhon.spectroscopy.
"""
import numpy as np

from typhon import spectroscopy


class TestSpectroscopy(object):
    """Testing the spectroscopy functions."""
    def test_linewidth(self):
        """Test calculation of full-width at half maximum."""
        f = np.linspace(0, np.pi, 100)
        a = np.cos(f)**2

        fwhm = spectroscopy.linewidth(f, a)

        assert np.allclose(fwhm, 1.5707963)
