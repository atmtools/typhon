# -*- coding: utf-8 -*-

"""Testing the basic geodetic functions.
"""

import numpy as np

from typhon import constants
from typhon import geodesy


class TestGeodesy(object):
    """Testing the geodetic functions."""
    def test_ellipsoidmodels(self):
        """Check ellipsoidmodels for valid excentricities."""
        e = geodesy.ellipsoidmodels()

        exc = np.array([e[m][1] for m in e.models])

        assert (np.all(exc >= 0) and np.all(exc < 1))

    def test_sind(self):
        """Test sinus calculation with degrees."""
        assert geodesy.sind(45) == np.sin(np.deg2rad(45))

    def test_cosd(self):
        """Test cosinus calculation with degrees."""
        assert geodesy.cosd(45) == np.cos(np.deg2rad(45))

    def test_tand(self):
        """Test tangens calculation with degrees."""
        assert geodesy.tand(45) == np.tan(np.deg2rad(45))

    def test_asind(self):
        """Test arcsinus calculation with degrees."""
        assert geodesy.asind(45) == np.arcsin(np.deg2rad(45))

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

        conversion = geodesy.cart2geocentric(*cartesian)

        assert np.allclose(conversion, reference)

    def test_geocentric2cart(self):
        """Test conversion from geocentric to cartesian system."""
        geocentric = (np.array([1, 1, 1]),  # r
                      np.array([0, 0, 90]),  # lat
                      np.array([0, 90, 0]),  # lon
                      )

        reference = (np.array([1, 0, 0]),  # x
                     np.array([0, 1, 0]),  # y
                     np.array([0, 0, 1]),  # z
                     )

        conversion = geodesy.geocentric2cart(*geocentric)

        assert np.allclose(conversion, reference)

    def test_geocentric2cart2geocentric(self):
        """Test conversion from geocentric to cartesian system and back."""
        ref = (1, -13, 42)

        cart = geodesy.geocentric2cart(*ref)
        geo = geodesy.cart2geocentric(*cart)

        assert np.allclose(ref, geo)

    def test_cart2geodetic(self):
        """Test conversion from cartesian to geodetic system."""
        r = geodesy.ellipsoidmodels().get('WGS84')[0]

        cartesian = (np.array([r, 0]),  # x
                     np.array([0, r]),  # y
                     np.array([0, 0]),  # z
                     )

        reference = (np.array([0, 0]),  # r
                     np.array([0, 00]),  # lat
                     np.array([0, 90]),  # lon
                     )

        conversion = geodesy.cart2geodetic(*cartesian)

        assert np.allclose(conversion, reference)

    def test_geodetic2cart(self):
        """Test conversion from geodetic to cartesian system."""
        r = geodesy.ellipsoidmodels().get('WGS84')[0]

        geodetic = (np.array([0, 0]),  # r
                    np.array([0, 0]),  # lat
                    np.array([0, 90]),  # lon
                    )

        reference = (np.array([r, 0]),  # x
                     np.array([0, r]),  # y
                     np.array([0, 0]),  # z
                     )

        conversion = geodesy.geodetic2cart(*geodetic)

        assert np.allclose(conversion, reference)

    def test_geodetic2cart2geodetic(self):
        """Test conversion from geodetic to cartesian system and back."""
        ellipsoid = geodesy.ellipsoidmodels().get('WGS84')
        ref = (1, -13, 42)

        cart = geodesy.geodetic2cart(*ref, ellipsoid)
        geod = geodesy.cart2geodetic(*cart, ellipsoid)

        assert np.allclose(ref, geod)

    def test_geodetic2geocentric2geodetic(self):
        """Test conversion from geodetic to geocentric system and back."""
        ellipsoid = geodesy.ellipsoidmodels().get('WGS84')
        ref = (1, -13, 42)

        geoc = geodesy.geodetic2geocentric(*ref, ellipsoid)
        geod = geodesy.geocentric2geodetic(*geoc, ellipsoid)

        assert np.allclose(ref, geod)

    def test_ellipsoid_r_geodetic(self):
        """Test return of geodetic radius for all ellipsois."""
        ellipsoid = geodesy.ellipsoidmodels().get('WGS84')

        r = geodesy.ellipsoid_r_geodetic(ellipsoid, 0)

        # Radius at equator has to be equal to the one defined in the
        # ellipsoidmodel.
        assert ellipsoid[0] == r

    def test_ellipsoid_r_geocentric(self):
        """Test return of geocentric radius for all ellipsois."""
        ellipsoid = geodesy.ellipsoidmodels().get('WGS84')

        r = geodesy.ellipsoid_r_geocentric(ellipsoid, 0)

        # Radius at equator has to be equal to the one defined in the
        # ellipsoidmodel.
        assert ellipsoid[0] == r

    def test_ellipsoid2d(self):
        """Test the calculation of new inclinated ellipsoids."""
        ellipsoid = geodesy.ellipsoidmodels().get('WGS84')

        e = geodesy.ellipsoid2d(ellipsoid, 0)

        assert e == (ellipsoid[0], 0)

    def test_ellipsoidcurvradius(self):
        """Test the calculation of local curvature radius."""
        ellipsoid = geodesy.ellipsoidmodels().get('WGS84')

        e = geodesy.ellipsoidcurvradius(ellipsoid, 0, 90)

        assert e == (ellipsoid[0], 0)

    def test_great_circle_distance(self):
        """Test calculation of great circle distance."""
        distance = geodesy.great_circle_distance(90, 30, 100, 60)

        assert distance == 10

    def test_great_circle_distance_radius(self):
        """Test calculation of great circle distance with radious."""
        r = constants.earth_radius

        distance = geodesy.great_circle_distance(0, 0, 180, 0, r=r)

        assert distance == np.pi * r

    def test_geogeraphic_mean(self):
        """Test calculation of geographical mean of coordinates."""
        mean = geodesy.geographic_mean(0, [89, 90, 91])

        assert mean == (0, 90)

    def test_cartposlos2geocentric(self):
        """Test conversion of cartesian POS/LOS to spherical coordinates."""
        # TODO: Consider to implement a test for intuitive values.
        # The current reference are just arbitrary chosen to check that the
        # behaviour in future implementations does not change.

        # Reference svn revision 10041.
        reference = (
            np.array([1.73205081, 1.8493242]),
            np.array([35.26438968, 36.49922243]),
            np.array([45., 42.27368901]),
            np.array([0., 2.53049304]),
            np.array([0., 118.40177722])
            )

        results = geodesy.cartposlos2geocentric(
            [1, 1.1], [1, 1], [1, 1.1], [1, 1], [1, 1], [1, 1])

        assert np.allclose(results, reference)
