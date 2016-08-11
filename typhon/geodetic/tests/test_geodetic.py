# -*- coding: utf-8 -*-

"""Testing the basic geodetic functions.
"""

import numpy as np

from typhon import geodetic


class TestConversion(object):
    """Testing the geodetic conversion functions.

    This class provides functions to test the conversion between different
    coordinate systems.
    """
    def test_cart2geocentric(self):
        """Test conversion from cartesian to geocentric system."""
        cartesians = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        references = [(1, 0, 0), (1, 0, 90), (1, 90, 0)]

        for c, r in zip(cartesians, references):
            yield self._convert_cart2geocentric, c, r

    def _convert_cart2geocentric(self, cart, reference):
        conversion = geodetic.cart2geocentric(*cart)
        assert np.allclose(conversion, reference)

    def test_geocentric2cart(self):
        """Test conversion from cartesian to geocentric system."""
        geocentrics = [(1, 0, 0), (1, 0, 90), (1, 90, 0)]
        references = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        for g, r in zip(geocentrics, references):
            yield self._convert_geocentric2cart, g, r

    def _convert_geocentric2cart(self, geocentric, reference):
        conversion = geodetic.geocentric2cart(*geocentric)
        assert np.allclose(conversion, reference)

    def test_geocentric2cart2geocentric(self):
        """Test conversion from geocentric to cartesian system and back."""
        ref = (1, -13, 42)

        cart = geodetic.geocentric2cart(*ref)
        geo = geodetic.cart2geocentric(*cart)

        assert np.allclose(ref, geo)
