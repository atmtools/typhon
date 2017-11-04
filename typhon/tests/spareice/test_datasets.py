from datetime import datetime

from typhon.spareice.datasets import Dataset


class TestDataset:
    """Testing the dataset methods."""
    def test_find_files(self):
        """Test finding files."""

        ds = Dataset(
            "reference/{year}/{month}/{day}/{hour}{minute}{second}-{end_hour}"
            "{end_minute}{end_second}.nc"
        )

        for file, times in ds.find_files(
                "2017-01-01", datetime(2017, 1, 2, 23), verbose=True):
            print(file)