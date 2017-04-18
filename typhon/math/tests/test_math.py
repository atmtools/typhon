# -*- coding: utf-8 -*-

"""Testing the basic mathematical functions.
"""

import numpy as np

from typhon import math


class TestCommon(object):
    """Testing common mathematical functions."""
    def test_integrate_column(self):
        """Test numerical intergration of array elements.

        As no coordinates are given, the function is expected to integrate
        along the indices.
        """
        t = math.integrate_column(np.arange(5))

        assert(np.isclose(t, 8))

    def test_integrate_column_coordinates(self):
        """Test numerical intergration of array elements along coordinates."""
        x = np.linspace(0, 1, 5)
        t = math.integrate_column(np.arange(5), x)

        assert(np.isclose(t, 2))

    def test_interpolate_halflevels(self):
        ref = np.array([0.5, 1.5, 2.5])
        t = math.interpolate_halflevels(np.arange(4))

        assert(np.allclose(t, ref))

    def test_sum_digits(self):
        """Test calculating several digit sums."""
        assert(math.sum_digits(1) == 1)
        assert(math.sum_digits(10) == 1)
        assert(math.sum_digits(100) == 1)
        assert(math.sum_digits(123) == 6)
        assert(math.sum_digits(2**8) == 13)

    def test_nlogspace(self):
        """Test creation of logarithmic spaced array."""
        ref = np.array([1, 10, 100])
        t = math.nlogspace(1, 100, 3)

        assert(np.allclose(t, ref))

    def test_promote_maximally(self):
        """Test copying to high precision dtype."""
        x = np.array(1, dtype='int8')
        t = math.promote_maximally(x)

        assert(t.dtype == 'int64')
