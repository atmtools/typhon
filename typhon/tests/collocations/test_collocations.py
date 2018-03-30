from datetime import datetime
from os.path import dirname, join
import tempfile

import numpy as np
from typhon.collections import DataGroup
from typhon.collocations import collocate, Collocations
from typhon.files import FileSet
import xarray as xr


class TestCollocations:
    """Testing the collocation functions."""

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
        return

        # Collect the data from all datasets and collocate them by once, should
        # give the same results when using collocate_datasets.
        a_dataset = Dataset(
            join(
                self.refdir,
                "tutorial_datasets/SatelliteA/{year}/{month}/{day}/{hour}"
                "{minute}{second}-{end_hour}{end_minute}{end_second}.nc.gz"
            ),
            name="SatelliteA",
        )
        b_dataset = Dataset(
            join(
                self.refdir,
                "tutorial_datasets/SatelliteB/{year}/{month}/{day}/{hour}"
                "{minute}{second}-{end_hour}{end_minute}{end_second}.nc.gz"
            ),
            name="SatelliteB",
        )

        # collocate_datasets creates new files that we do not want to keep.
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create the output dataset:
            ab_collocations = Dataset(
                path=join(
                    tmpdirname,
                    "{year}/{month}/{day}/{hour}{minute}{second}-"
                    "{end_hour}{end_minute}{end_second}.nc"
                )
            )

            self._test_collocate_datasets(
                a_dataset, b_dataset, ab_collocations
            )

    @staticmethod
    def _test_collocate_datasets(a_dataset, b_dataset, ab_collocations):

        start, end = "2018-01-01", datetime(2018, 1, 2)

        # Collected and collocated all at once:
        a_data_all = a_dataset.collect(start, end)
        b_data_all = b_dataset.collect(start, end)
        pairs = collocate([a_data_all, b_data_all], max_interval="4h",
                          max_distance="300km")
        a_reference = a_data_all[pairs[0]]
        b_reference = b_data_all[pairs[1]]

        # Using collocate_datasets
        collocate_datasets(
            [a_dataset, b_dataset], start=start, end=end,
            output=ab_collocations, max_interval="4h", max_distance="300km"
        )

        a_retrieved, b_retrieved = None, None
        for data in ab_collocations.icollect(start, end):
            pairs = data["__collocations/SatelliteA.SatelliteB/pairs"]

            del data["SatelliteA/__file_start"]
            del data["SatelliteA/__file_end"]
            del data["SatelliteA/__indices"]
            del data["SatelliteB/__file_start"]
            del data["SatelliteB/__file_end"]
            del data["SatelliteB/__indices"]

            if a_retrieved is None:
                a_retrieved = data["SatelliteA"][pairs[0]]
                b_retrieved = data["SatelliteB"][pairs[1]]
            else:
                a_retrieved = GroupedArrays.concat(
                    [a_retrieved, data["SatelliteA"][pairs[0]]])
                b_retrieved = GroupedArrays.concat(
                    [b_retrieved, data["SatelliteB"][pairs[1]]])

        assert a_reference == a_retrieved
        assert b_reference == b_retrieved
