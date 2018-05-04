# -*- coding: utf-8 -*-
"""Testing the functions in typhon.geographical.
"""
import numpy as np

from typhon import geographical


class TestGeographical:
    """Testing the geographical functions."""
    def test_area_weighted_mean(self):
        """Test calculation of area weighted mean."""
        lon = np.arange(0, 360, 10)
        lat = np.arange(-90, 90, 10)
        f = np.ones(lon.shape) + np.cos(np.deg2rad(lat))[:, np.newaxis]
        mean = geographical.area_weighted_mean(lon, lat, f)

        assert np.allclose(mean, 1.7852763105888174)


class TestGeoIndex:
    """Testing the GeoIndex functions."""

    def test_query(self):
        """Test the function collocate with dicts, xarray and GroupedArrays.
        Checks on temporal, spatial and temporal-spatial conditions.

        """
        # Update this test
        return

        # Test with dictionaries:
        lat1 = 30. * np.sin(np.linspace(-3.14, 3.14, 24)) + 20
        lon1 = np.linspace(0, 90, 24)
        lat2 = 30. * np.sin(np.linspace(-3.14, 3.14, 24) + 1.) + 20
        lon2 = np.linspace(0, 90, 24)

        index = geographical.GeoIndex(lat1, lon1)
        pairs, distances = index.query(lat2, lon2, r="500 km")

        check_pairs = [
            [4, 15, 15, 16],
            [4, 15, 16, 15]
        ]

        assert pairs.tolist() == check_pairs
