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
    def test_ellipsoidmodels(self):
        """Check ellipsoidmodels for valid excentricities."""
        e = geodetic.ellipsoidmodels()

        exc = np.array([e[m][1] for m in e.models])

        assert (np.all(exc >= 0) and np.all(exc < 1))

    def test_cart2geocentric(self):
        """Test conversion from cartesian to geocentric system."""
        cartesian = (np.array([1, 0, 0]),  # x
                     np.array([0, 1, 0]),  # y
                     np.array([0, 0, 1]),  # z
                     )

        reference = (np.array([1, 1, 1]),  # r
                     np.array([0, 0, 90]),  # lat
                     np.array([0, 90, 0]),  # lon
                     )

        conversion = geodetic.cart2geocentric(*cartesian)

        assert np.allclose(conversion, reference)

    def test_geocentric2cart(self):
        """Test conversion from cartesian to geocentric system."""
        geocentric = (np.array([1, 1, 1]),  # r
                      np.array([0, 0, 90]),  # lat
                      np.array([0, 90, 0]),  # lon
                      )

        reference = (np.array([1, 0, 0]),  # x
                     np.array([0, 1, 0]),  # y
                     np.array([0, 0, 1]),  # z
                     )

        conversion = geodetic.geocentric2cart(*geocentric)

        assert np.allclose(conversion, reference)

    def test_geocentric2cart2geocentric(self):
        """Test conversion from geocentric to cartesian system and back."""
        ref = (1, -13, 42)

        cart = geodetic.geocentric2cart(*ref)
        geo = geodetic.cart2geocentric(*cart)

        assert np.allclose(ref, geo)

    def test_geodetic2cart2geodetic(self):
        """Test geodetic/cartesian conversion for all ellipsoids."""
        e = geodetic.ellipsoidmodels()

        for model in e.models:
            yield self._test_geodetic2cart2geodetic, e[model]

    def _test_geodetic2cart2geodetic(self, ellipsoid):
        """Test conversion from geodetic to cartesian system and back."""
        ref = (1, -13, 42)

        cart = geodetic.geodetic2cart(*ref, ellipsoid)
        geod = geodetic.cart2geodetic(*cart, ellipsoid)

        assert np.allclose(ref, geod)

    def test_geodetic2geocentric2geodetic(self):
        """Test geodetic/geocentric conversion for all ellipsoids."""
        e = geodetic.ellipsoidmodels()

        for model in e.models:
            yield self._test_geodetic2geocentric2geodetic, e[model]

    def _test_geodetic2geocentric2geodetic(self, ellipsoid):
        """Test conversion from geodetic to geocentric system and back."""
        ref = (1, -13, 42)

        geoc = geodetic.geodetic2geocentric(*ref, ellipsoid)
        geod = geodetic.geocentric2geodetic(*geoc, ellipsoid)

        assert np.allclose(ref, geod)
