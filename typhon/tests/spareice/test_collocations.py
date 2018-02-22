from os.path import dirname, join

import numpy as np
from typhon.spareice import collocate, collocate_datasets
from typhon.spareice import Dataset, GroupedArrays
import xarray as xr


class TestCollocations:
    """Testing the dataset methods."""

    datasets = None
    refdir = join(dirname(__file__), 'reference')

    def test_collocate(self):
        """Test the function collocate with dicts, xarray and GroupedArrays.
        Checks on temporal, spatial and temporal-spatial conditions.

        """

        # Test with dictionaries:
        primary = {
            "time": np.arange("2018-01-01", "2018-01-02",
                              dtype="datetime64[h]"),
            "lat": 30. * np.sin(np.linspace(-3.14, 3.14, 24)) + 20,
            "lon": np.linspace(0, 90, 24),
        }
        secondary = {
            "time": np.arange("2018-01-01", "2018-01-02",
                              dtype="datetime64[h]"),
            "lat": 30. * np.sin(np.linspace(-3.14, 3.14, 24) + 1.) + 20,
            "lon": np.linspace(0, 90, 24),
        }
        self._test_collocate(primary, secondary)

        # Test with xarray.Dataset:
        self._test_collocate(xr.Dataset(primary), xr.Dataset(secondary))

        # Test with GroupedArrays:
        self._test_collocate(
            GroupedArrays.from_dict(primary),
            GroupedArrays.from_dict(secondary)
        )

    @staticmethod
    def _test_collocate(primary, secondary):
        check_all = [[4, 15], [4, 15]]
        check_spatial = [
            [4, 15, 15, 16],
            [4, 15, 16, 15]
        ]
        check_temporal = [
            [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,
             6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11,
             11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16,
             17,
             17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22,
             22, 23, 23],
            [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6,
             5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10, 9, 10, 11, 10, 11,
             12, 11, 12, 13, 12, 13, 14, 13, 14, 15, 14, 15, 16, 15, 16, 17,
             16, 17, 18, 17, 18, 19, 18, 19, 20, 19, 20, 21, 20, 21, 22, 21,
             22, 23, 22, 23]
        ]

        assert collocate(
            [primary, secondary], max_distance="500km", max_interval="1h",
        ).tolist() == check_all
        assert collocate(
            [primary, secondary], max_interval="1h",
        ).tolist() == check_temporal
        assert collocate(
            [primary, secondary], max_distance="500km",
        ).tolist() == check_spatial

    def test_collocate_datasets(self):
        # TODO: Implement test for collocate_datasets
        return
