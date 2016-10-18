# -*- coding: utf-8 -*-

"""Testing the functions in typhon.geographical.
"""

import numpy as np

from typhon import geographical


class TestGeographical(object):
    """Testing the geographical functions."""
    def test_area_weighted_mean(self):
        """Test calculation of area weighted mean."""
        lon = np.arange(0, 360, 10)
        lat = np.arange(-90, 90, 10)
        f = np.ones(lon.shape) + np.cos(np.deg2rad(lat))[:, np.newaxis]
        mean = geographical.area_weighted_mean(lon, lat, f)

        assert mean == 1.7852763105888174
