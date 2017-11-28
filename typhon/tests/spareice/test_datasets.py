from datetime import datetime
from os.path import (dirname, join)

from typhon.spareice.datasets import Dataset


class TestDataset:
    """Testing the dataset methods."""

    refdir = join(dirname(__file__), 'reference')

    def test_contains(self):
        refpattern = (
            "{year}/{month}/{day}/"
            "{hour}{minute}{second}-{end_hour}{end_minute}{end_second}.nc"
        )
        dataset_files = join(self.refdir, refpattern)
        ds = Dataset(dataset_files)

        assert (
            datetime(2016, 1, 1) not in ds
            and datetime(2017, 1, 1) in ds
            and "2016-01-01 00:00:00" not in ds
            and "2017-01-01 00:00:00" in ds
            and ("2017-01-01", "2017-01-02") in ds
        )

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

        found_file = ds.find_file("2017-01-04 13:00:00")
        file = self.refdir + "/2017/01/03/180000-000000.nc"
        assert found_file == file

    def test_find_files1(self):
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
            [join(self.refdir, '2017/01/01/000000-060000.nc'), (
                datetime(2017, 1, 1, 0, 0),
                datetime(2017, 1, 1, 6, 0))], [
             join(self.refdir, '2017/01/01/060000-120000.nc'), (
                 datetime(2017, 1, 1, 6, 0),
                 datetime(2017, 1, 1, 12, 0))], [
             join(self.refdir, '2017/01/01/120000-180000.nc'), (
                 datetime(2017, 1, 1, 12, 0),
                 datetime(2017, 1, 1, 18, 0))], [
             join(self.refdir, '2017/01/01/180000-000000.nc'), (
                 datetime(2017, 1, 1, 18, 0),
                 datetime(2017, 1, 2, 0, 0))], [
             join(self.refdir, '2017/01/02/000000-060000.nc'), (
                 datetime(2017, 1, 2, 0, 0),
                 datetime(2017, 1, 2, 6, 0))], [
             join(self.refdir, '2017/01/02/060000-120000.nc'), (
                 datetime(2017, 1, 2, 6, 0),
                 datetime(2017, 1, 2, 12, 0))], [
             join(self.refdir, '2017/01/02/120000-180000.nc'), (
                 datetime(2017, 1, 2, 12, 0),
                 datetime(2017, 1, 2, 18, 0))], [
             join(self.refdir, '2017/01/02/180000-000000.nc'), (
                 datetime(2017, 1, 2, 18, 0),
                 datetime(2017, 1, 3, 0, 0))]
        ]

        assert found_files == files

    def test_find_files2(self):
        ds1 = Dataset(
            join(self.refdir,
                 "{year}/{month}/{day}/{hour}{minute}{second}-{end_hour}"
                 "{end_minute}{end_second}.nc"
                 )
        )

        found_files = list(ds1.find_files("2017-01-01 18:00:00",
                                          "2017-01-02 08:00:00"))

        files = [
            [join(self.refdir, '2017/01/01/120000-180000.nc'),
             (datetime(2017, 1, 1, 12, 0), datetime(2017, 1, 1, 18, 0))],
            [join(self.refdir, '2017/01/01/180000-000000.nc'),
             (datetime(2017, 1, 1, 18, 0), datetime(2017, 1, 2, 0, 0))],
            [join(self.refdir, '2017/01/02/000000-060000.nc'),
             (datetime(2017, 1, 2, 0), datetime(2017, 1, 2, 6))],
            [join(self.refdir, '2017/01/02/060000-120000.nc'),
             (datetime(2017, 1, 2, 6, 0), datetime(2017, 1, 2, 12, 0))]
        ]

        assert found_files == files

    def test_find_files_with_wildcards(self):
        """Test finding files."""

        refpattern = (
            "{year}/{month}/{day}/{hour}{minute}{second}*.nc"
        )
        dataset_files = join(self.refdir, refpattern)
        ds = Dataset(dataset_files)

        found_files = list(
            ds.find_files("2017-01-01", "2017-01-02 23:00:00", sort=True)
        )

        files = [
            [join(self.refdir, '2017/01/01/000000-060000.nc'), (
                datetime(2017, 1, 1, 0, 0),
                datetime(2017, 1, 1, 0, 0))], [
             join(self.refdir, '2017/01/01/060000-120000.nc'), (
                 datetime(2017, 1, 1, 6, 0),
                 datetime(2017, 1, 1, 6, 0))], [
             join(self.refdir, '2017/01/01/120000-180000.nc'), (
                 datetime(2017, 1, 1, 12, 0),
                 datetime(2017, 1, 1, 12, 0))], [
             join(self.refdir, '2017/01/01/180000-000000.nc'), (
                 datetime(2017, 1, 1, 18, 0),
                 datetime(2017, 1, 1, 18, 0))], [
             join(self.refdir, '2017/01/02/000000-060000.nc'), (
                 datetime(2017, 1, 2, 0, 0),
                 datetime(2017, 1, 2, 0, 0))], [
             join(self.refdir, '2017/01/02/060000-120000.nc'), (
                 datetime(2017, 1, 2, 6, 0),
                 datetime(2017, 1, 2, 6, 0))], [
             join(self.refdir, '2017/01/02/120000-180000.nc'), (
                 datetime(2017, 1, 2, 12, 0),
                 datetime(2017, 1, 2, 12, 0))], [
             join(self.refdir, '2017/01/02/180000-000000.nc'), (
                 datetime(2017, 1, 2, 18, 0),
                 datetime(2017, 1, 2, 18, 0))]
        ]

        assert found_files == files

    def test_retrieve_time_coverage_with_wildcards(self):
        ds1 = Dataset(
            join(self.refdir,
                 "plain_dataset/file-{year}{month}{day}{hour}{minute}{second}_"
                 "*{end_hour}{end_minute}{end_second}.nc")
        )

        found_start, found_end = ds1.retrieve_time_coverage(
            join(self.refdir,
                 'plain_dataset/file-20170101120000_20170102000000.nc')
        )

        start = datetime(2017, 1, 1, 12)
        end = datetime(2017, 1, 2)

        assert found_start == start and found_end == end

    def test_find_files_plain(self):
        ds1 = Dataset(
            join(self.refdir,
                 "plain_dataset/file-{year}{month}{day}{hour}{minute}{second}_"
                 "{end_year}{end_month}{end_day}{end_hour}{end_minute}"
                 "{end_second}.nc")
        )

        found_files = list(ds1.find_files(
            "2017-01-01 18:00:00", "2017-01-02 08:00:00", sort=True
        ))

        files = [
            [join(self.refdir,
                  'plain_dataset/file-20170101120000_20170102000000.nc'),
             (datetime(2017, 1, 1, 12, 0),
              datetime(2017, 1, 2, 0, 0))],
            [join(self.refdir,
                  'plain_dataset/file-20170102000000_20170102120000.nc'),
             (datetime(2017, 1, 2, 0, 0),
              datetime(2017, 1, 2, 12, 0))]
        ]

        assert found_files == files

    def test_find_files_single(self):
        ds1 = Dataset(
            join(self.refdir, "dataset_of_single_file.nc")
        )

        found_files = list(ds1.find_files(
                "2017-01-01 18:00:00", "2017-01-02 08:00:00"))

        file = join(self.refdir, "dataset_of_single_file.nc")

        assert len(found_files) == 1
        assert found_files[0][0] == file

    def test_find_overlapping_files(self):
        # So far this test does not work due to ordering problems.
        pass

        # ds1 = Dataset(
        #     join(self.refdir,
        #          "{year}/{month}/{day}/{hour}{minute}{second}-{end_hour}"
        #          "{end_minute}{end_second}.nc"
        #          )
        # )
        #
        # ds2 = Dataset(
        #     join(self.refdir,
        #          "{year}/{month}/{doy}/{hour}{minute}{second}-{end_hour}"
        #          "{end_minute}{end_second}.nc"
        #          )
        # )
        #
        # overlapping_files = list(ds1.find_overlapping_files(
        #     "2017-01-01 18:00:00", "2017-01-02 08:00:00", ds2))
        #
        # for file in overlapping_files:
        #     print(file)
        #
        # files = [
        #     (join(self.refdir, '2017/01/01/120000-180000.nc'),
        #      [join(self.refdir, '2017/01/001/120000-180000.nc'),
        #       join(self.refdir, '2017/01/001/180000-000000.nc'),
        #       join(self.refdir, '2017/01/002/000000-060000.nc')]),
        #     (join(self.refdir, '2017/01/01/180000-000000.nc'),
        #      [join(self.refdir, '2017/01/001/120000-180000.nc'),
        #       join(self.refdir, '2017/01/001/180000-000000.nc')]),
        #     (join(self.refdir, '2017/01/01/000000-060000.nc'),
        #      [join(self.refdir, '2017/01/002/120000-180000.nc'),
        #       join(self.refdir, '2017/01/002/180000-000000.nc')]),
        # ]
        #
        # assert overlapping_files == files
