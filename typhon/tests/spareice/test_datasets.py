from datetime import datetime
from os.path import dirname, join

from typhon.spareice.datasets import Dataset, DatasetManager
from typhon.spareice.handlers import FileHandler, FileInfo


class TestDataset:
    """Testing the dataset methods."""

    refdir = join(dirname(__file__), 'reference')
    time_periods = [
        [],
    ]

    def load_datasets(self):
        dataset = DatasetManager()

        dataset += Dataset(
            join(
                self.refdir,
                "{year}/{month}/{day}/{hour}{minute}{second}-{end_hour}{end_minute}{end_second}.nc",  # noqa
            ),
            name="start-end",
        )
        dataset += Dataset(
            join(
                self.refdir, "{year}/{month}/{day}/{hour}{minute}{second}*.nc",
            ),
            name="start-wildcard",
        )
        dataset += Dataset(
            join(self.refdir, "dataset_of_single_file.nc",),
            name="single-file",
            time_coverage=["2017-01-01", "2017-01-03"],
        )

        def sequence_get_info(filename, **kwargs):
            """Small helper function for sequence dataset."""
            info = FileInfo(filename)
            with open(filename) as f:
                info.times[0] = datetime.strptime(
                    f.readline().rstrip(),
                    "Start: %Y-%m-%d %H:%M:%S"
                )
                info.times[1] = datetime.strptime(
                    f.readline().rstrip(),
                    "End: %Y-%m-%d %H:%M:%S"
                )
            return info

        dataset += Dataset(
            join(self.refdir, "sequence_dataset/{year}/{doy}/sequence*.txt",),
            name="sequence",
            handler=FileHandler(
                info_reader=sequence_get_info,
            ),
            time_coverage="content",
        )

        return dataset

    def test_contains(self):
        """Test whether all datasets cover the testing timestamps.

        Returns:
            None
        """
        datasets = self.load_datasets()
        tests = [
            # [Timestamp(s), Should it be covered by the datasets?]
            ["2016-01-01", False],
            ["2017-01-01", True],
            ["2017-01-02", True],
            [datetime(2017, 1, 1), True],
            [datetime(2017, 1, 2, 12), True],
        ]

        for name, dataset in datasets.items():
            print("Run test-contains for %s dataset:" % name)

            for timestamp, check in tests:
                print("\tCheck whether %s is covered (expected %s, got %s)" % (
                    timestamp, check, timestamp in dataset
                ))
                assert (timestamp in dataset) == check

    def test_find_files(self):
        """Test whether all datasets cover the testing timestamps.

        TODO:
            Extend this test.

        Returns:
            None
        """
        datasets = self.load_datasets()
        tests = [
            # [Time period, Files that should be found (1)]
            # (1) Empty list means that no files should be found
            [("2016-01-01", "2016-01-01"), []],
            #[("2017-01-01", "2017-01-02"), ],
        ]

        for name, dataset in datasets.items():
            print("Run test-find_files for %s dataset:" % name)

            for period, check in tests:
                result = list(
                    dataset.find_files(*period, no_files_error=False))

                print("\tCheck whether %s is covered (expected %s, got %s)" % (
                    period, bool(check), bool(result),
                ))

                assert result == check
