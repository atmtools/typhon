# -*- coding: utf-8 -*-
"""Testing the basic mathematical functions.
"""
import numpy as np

import pytest

from typhon import math


class TestCommon:
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

    def test_squeezable_logspace(self):
        """Test creation of squeezable logspace."""
        ref = np.array([1, 3.16227766, 100])
        t = math.squeezable_logspace(1, 100, 3, squeeze=0.5)

        assert(np.allclose(t, ref))

    def test_squeezable_logspace_nosqueeze(self):
        """Test creation of non-squeezed logspace.

        Without squeezing, results should be equal to normal ``nlogspace``.
        """
        ref = np.array([1, 10, 100])
        t = math.squeezable_logspace(1, 100, 3)

        assert(np.allclose(t, ref))

    def test_squeezable_logspace_fixpointbounds(self):
        """Test ValueError if fixpoint is out of bounds."""
        with pytest.raises(ValueError):
            math.squeezable_logspace(100, 1, fixpoint=1.1)

    def test_squeezable_logspace_squeezebounds(self):
        """Test ValueError if squeeze is out of bounds."""
        with pytest.raises(ValueError):
            math.squeezable_logspace(100, 1, squeeze=2.01)

class TestArray:
    """Testing array related mathematical functions."""
    def test_argclosest(self):
        x = np.arange(10)

        idx = math.array.argclosest(x, value=5.4)
        assert idx == 5

        idx, val = math.array.argclosest(x, value=5.4, retvalue=True)
        assert idx == 5 and val == 5.
