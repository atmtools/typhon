from datetime import datetime
from os.path import (dirname, join)

from typhon.spareice.datasets import Dataset


class TestDataset:
    """Testing the dataset methods."""

    refdir = join(dirname(__file__), 'reference')

    def test_find_file(self):
        refpattern = (
            "{year}/{month}/{day}/"
            "{hour}{minute}{second}-{end_hour}{end_minute}{end_second}.nc"
        )
        dataset_files = join(self.refdir, refpattern)
        ds = Dataset(dataset_files)

        found_file = ds.find_file("2017-01-01 10:00:00")
        file = self.refdir + "/2017/01/01/060000-120000.nc"

        assert found_file == file

    def test_find_files(self):
        """Test finding files."""
        refpattern = (
            "{year}/{month}/{day}/"
            "{hour}{minute}{second}-{end_hour}{end_minute}{end_second}.nc"
        )
        dataset_files = join(self.refdir, refpattern)

        ds = Dataset(dataset_files)

        found_files = list(
            ds.find_files("2017-01-01", datetime(2017, 1, 2, 23), sort=True)
        )

        files = [
            [join(self.refdir, '2017/01/01/000000-060000.nc'), [
                datetime(2017, 1, 1, 0, 0),
                datetime(2017, 1, 1, 6, 0)]], [
             join(self.refdir, '2017/01/01/060000-120000.nc'), [
                 datetime(2017, 1, 1, 6, 0),
                 datetime(2017, 1, 1, 12, 0)]], [
             join(self.refdir, '2017/01/01/120000-180000.nc'), [
                 datetime(2017, 1, 1, 12, 0),
                 datetime(2017, 1, 1, 18, 0)]], [
             join(self.refdir, '2017/01/01/180000-000000.nc'), [
                 datetime(2017, 1, 1, 18, 0),
                 datetime(2017, 1, 2, 0, 0)]], [
             join(self.refdir, '2017/01/02/000000-060000.nc'), [
                 datetime(2017, 1, 2, 0, 0),
                 datetime(2017, 1, 2, 6, 0)]], [
             join(self.refdir, '2017/01/02/060000-120000.nc'), [
                 datetime(2017, 1, 2, 6, 0),
                 datetime(2017, 1, 2, 12, 0)]], [
             join(self.refdir, '2017/01/02/120000-180000.nc'), [
                 datetime(2017, 1, 2, 12, 0),
                 datetime(2017, 1, 2, 18, 0)]], [
             join(self.refdir, '2017/01/02/180000-000000.nc'), [
                 datetime(2017, 1, 2, 18, 0),
                 datetime(2017, 1, 3, 0, 0)]]
        ]

        print(files)
        print(found_files)

        assert found_files == files

    def test_find_files_with_wildcards(self):
        """Test finding files."""

        refpattern = (
            "{year}/{month}/{day}/{hour}{minute}{second}*.nc"
        )
        dataset_files = join(self.refdir, refpattern)
        ds = Dataset(dataset_files)

        found_files = list(
            ds.find_files("2017-01-01", datetime(2017, 1, 2, 23), sort=True)
        )

        files = [
            [join(self.refdir, '2017/01/01/000000-060000.nc'), [
                datetime(2017, 1, 1, 0, 0),
                datetime(2017, 1, 1, 0, 0)]], [
             join(self.refdir, '2017/01/01/060000-120000.nc'), [
                 datetime(2017, 1, 1, 6, 0),
                 datetime(2017, 1, 1, 6, 0)]], [
             join(self.refdir, '2017/01/01/120000-180000.nc'), [
                 datetime(2017, 1, 1, 12, 0),
                 datetime(2017, 1, 1, 12, 0)]], [
             join(self.refdir, '2017/01/01/180000-000000.nc'), [
                 datetime(2017, 1, 1, 18, 0),
                 datetime(2017, 1, 1, 18, 0)]], [
             join(self.refdir, '2017/01/02/000000-060000.nc'), [
                 datetime(2017, 1, 2, 0, 0),
                 datetime(2017, 1, 2, 0, 0)]], [
             join(self.refdir, '2017/01/02/060000-120000.nc'), [
                 datetime(2017, 1, 2, 6, 0),
                 datetime(2017, 1, 2, 6, 0)]], [
             join(self.refdir, '2017/01/02/120000-180000.nc'), [
                 datetime(2017, 1, 2, 12, 0),
                 datetime(2017, 1, 2, 12, 0)]], [
             join(self.refdir, '2017/01/02/180000-000000.nc'), [
                 datetime(2017, 1, 2, 18, 0),
                 datetime(2017, 1, 2, 18, 0)]]
        ]

        print(files)
        print(found_files)

        assert found_files == files
