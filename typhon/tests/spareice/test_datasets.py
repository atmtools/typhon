import datetime
from os.path import dirname, join

from typhon.spareice.datasets import Dataset, DatasetManager
from typhon.spareice.handlers import FileHandler, FileInfo


class TestDataset:
    """Testing the dataset methods."""

    datasets = None
    refdir = join(dirname(__file__), 'reference')

    def init_datasets(self):
        if self.datasets is not None:
            return self.datasets

        self.datasets = DatasetManager()

        self.datasets += Dataset(
            join(
                self.refdir,
                "{year}/{month}/{day}/{hour}{minute}{second}-{end_hour}{end_minute}{end_second}.nc",  # noqa
            ),
            name="standard",
        )
        self.datasets += Dataset(
            join(
                self.refdir, "{year}/{month}/{day}/{hour}{minute}{second}*.nc",
            ),
            name="standard-wildcard",
        )
        self.datasets += Dataset(
            join(self.refdir, "dataset_of_single_file.nc",),
            name="single",
            time_coverage=["2017-01-01", "2017-01-03"],
        )

        def sequence_get_info(filename, **kwargs):
            """Small helper function for sequence dataset."""
            info = FileInfo(filename)
            with open(filename) as f:
                info.times[0] = datetime.datetime.strptime(
                    f.readline().rstrip(),
                    "Start: %Y-%m-%d %H:%M:%S"
                )
                info.times[1] = datetime.datetime.strptime(
                    f.readline().rstrip(),
                    "End: %Y-%m-%d %H:%M:%S"
                )
            return info

        self.datasets += Dataset(
            join(self.refdir, "sequence_dataset/{year}/{doy}/sequence*.txt",),
            name="sequence-wildcard",
            handler=FileHandler(
                info_reader=sequence_get_info,
            ),
            info_via="handler",
        )
        self.datasets += Dataset(
            join(
                self.refdir, "sequence_dataset/{year}/{doy}/sequence{id}.txt",
            ),
            handler=FileHandler(
                info_reader=sequence_get_info,
            ),
            name="sequence-placeholder",
            info_via="both",
        )
        self.datasets["sequence-placeholder"].placeholder["id"] = "(\d{4})"

        return self.datasets

    def test_contains(self):
        """Test whether all datasets cover the testing timestamps.

        Returns:
            None
        """
        datasets = self.init_datasets()
        tests = [
            # [Timestamp(s), Should it be covered by the datasets?]
            ["2016-01-01", False],
            ["2017-01-01", True],
            ["2017-01-02", True],
            [datetime.datetime(2017, 1, 1), True],
            [datetime.datetime(2017, 1, 2, 12), True],
        ]

        for name, dataset in datasets.items():
            print("Run test-contains for %s dataset:" % name)

            for timestamp, check in tests:
                print("\tCheck whether %s is covered (expected %s, got %s)" % (
                    timestamp, check, timestamp in dataset
                ))
                assert (timestamp in dataset) == check

    def test_find_files_standard(self):
        """Test find_files on the standard dataset.

        Returns:
            None
        """
        datasets = self.init_datasets()

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            datasets["standard"].find_files(
                "2016-12-31", "2017-01-01", no_files_error=False
            ))
        assert not empty

        # Should find four files:
        found_files = list(
            datasets["standard"].find_files(
                "2017-01-01", "2017-01-02",
            ))

        check = [
            FileInfo(join(self.refdir, '2017/01/01/000000-060000.nc'),
                     [datetime.datetime(2017, 1, 1, 0, 0),
                      datetime.datetime(2017, 1, 1, 6, 0)], {}),
            FileInfo(join(self.refdir, '2017/01/01/060000-120000.nc'),
                     [datetime.datetime(2017, 1, 1, 6, 0),
                      datetime.datetime(2017, 1, 1, 12, 0)], {}),
            FileInfo(join(self.refdir, '2017/01/01/120000-180000.nc'),
                     [datetime.datetime(2017, 1, 1, 12, 0),
                      datetime.datetime(2017, 1, 1, 18, 0)], {}),
            FileInfo(join(self.refdir, '2017/01/01/180000-000000.nc'),
                     [datetime.datetime(2017, 1, 1, 18, 0),
                      datetime.datetime(2017, 1, 2, 0, 0)], {}),

        ]

        assert found_files == check

        # Should find four files and should return them in two bins:
        found_files = list(
            datasets["standard"].find_files(
                "2017-01-01", "2017-01-02", bundle="12h",
            ))

        check = [
            [
                FileInfo(join(self.refdir, '2017/01/01/000000-060000.nc'),
                         [datetime.datetime(2017, 1, 1, 0, 0),
                          datetime.datetime(2017, 1, 1, 6, 0)], {}),
                FileInfo(join(self.refdir, '2017/01/01/060000-120000.nc'),
                         [datetime.datetime(2017, 1, 1, 6, 0),
                          datetime.datetime(2017, 1, 1, 12, 0)], {}),
            ],
            [
                FileInfo(join(self.refdir, '2017/01/01/120000-180000.nc'),
                         [datetime.datetime(2017, 1, 1, 12, 0),
                          datetime.datetime(2017, 1, 1, 18, 0)], {}),
                FileInfo(join(self.refdir, '2017/01/01/180000-000000.nc'),
                         [datetime.datetime(2017, 1, 1, 18, 0),
                          datetime.datetime(2017, 1, 2, 0, 0)], {}),
            ],
        ]

        assert found_files == check

        # Should find four files and should return them in two bins:
        found_files = list(
            datasets["standard"].find_files(
                "2017-01-01", "2017-01-02", bundle=3,
            ))

        check = [
            [
                FileInfo(join(self.refdir, '2017/01/01/000000-060000.nc'),
                         [datetime.datetime(2017, 1, 1, 0, 0),
                          datetime.datetime(2017, 1, 1, 6, 0)], {}),
                FileInfo(join(self.refdir, '2017/01/01/060000-120000.nc'),
                         [datetime.datetime(2017, 1, 1, 6, 0),
                          datetime.datetime(2017, 1, 1, 12, 0)], {}),
                FileInfo(join(self.refdir, '2017/01/01/120000-180000.nc'),
                         [datetime.datetime(2017, 1, 1, 12, 0),
                          datetime.datetime(2017, 1, 1, 18, 0)], {}),
            ],
            [
                FileInfo(join(self.refdir, '2017/01/01/180000-000000.nc'),
                         [datetime.datetime(2017, 1, 1, 18, 0),
                          datetime.datetime(2017, 1, 2, 0, 0)], {}),
            ],
        ]

        assert found_files == check

    def test_find_files_single(self):
        """Test find_files on the single dataset.

        Returns:
            None
        """
        datasets = self.init_datasets()

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            datasets["single"].find_files(
                "2016-12-31", "2017-01-01", no_files_error=False
            ))
        assert not empty

        check = [
            FileInfo(join(self.refdir, 'dataset_of_single_file.nc'),
                     [datetime.datetime(2017, 1, 1, 0, 0),
                      datetime.datetime(2017, 1, 3, 0, 0)], {}),
        ]

        found_files = list(
            datasets["single"].find_files(
                "2017-01-01", "2017-01-02",
            ))

        assert found_files == check

        found_files = list(
            datasets["single"].find_files(
                "2017-01-01", "2017-01-02", bundle="12h",
            ))

        assert found_files == check

        found_files = list(
            datasets["single"].find_files(
                "2017-01-01", "2017-01-02", bundle=3,
            ))

        assert found_files == check

    def test_find_files_sequence(self):
        """Test find_files on the sequence datasets.

        Returns:
            None
        """
        datasets = self.init_datasets()

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            datasets["sequence-placeholder"].find_files(
                "2016-12-31", "2017-01-01", no_files_error=False
            ))
        assert not empty

        # Should find two files:
        found_files = list(
            datasets["sequence-placeholder"].find_files(
                "2017-01-01", "2017-01-02",
            ))

        check = [
            FileInfo(join(self.refdir,
                          'sequence_dataset/2017/001/sequence0001.txt'),
                     [datetime.datetime(2017, 1, 1, 0, 0),
                      datetime.datetime(2017, 1, 1, 12, 0)], {'id': 1}),
            FileInfo(join(self.refdir,
                          'sequence_dataset/2017/001/sequence0002.txt'),
                     [datetime.datetime(2017, 1, 1, 12, 0),
                      datetime.datetime(2017, 1, 2, 0, 0)], {'id': 2}),

        ]
        assert found_files == check

        # Should find two files and should return them in two bins:
        found_files = list(
            datasets["sequence-placeholder"].find_files(
                "2017-01-01", "2017-01-02", bundle="6h",
            ))

        check = [
            [
                FileInfo(join(self.refdir,
                              'sequence_dataset/2017/001/sequence0001.txt'),
                         [datetime.datetime(2017, 1, 1, 0, 0),
                          datetime.datetime(2017, 1, 1, 12, 0)], {'id': 1}),
            ],
            [
                FileInfo(join(self.refdir,
                              'sequence_dataset/2017/001/sequence0002.txt'),
                         [datetime.datetime(2017, 1, 1, 12, 0),
                          datetime.datetime(2017, 1, 2, 0, 0)], {'id': 2}),
            ],
        ]
        assert found_files == check

    def test_find_files_sequence_placeholder(self):
        """Test find_files on all standard datasets.

        Returns:
            None
        """
        datasets = self.init_datasets()

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            datasets["sequence-placeholder"].find_files(
                "2016-12-31", "2017-01-01", no_files_error=False
            ))
        assert not empty

        # Should find two files:
        found_files = list(
            datasets["sequence-placeholder"].find_files(
                "2017-01-01", "2017-01-02",
            ))

        check = [
            FileInfo(join(self.refdir,
                          'sequence_dataset/2017/001/sequence0001.txt'),
                     [datetime.datetime(2017, 1, 1, 0, 0),
                      datetime.datetime(2017, 1, 1, 12, 0)], {'id': 1}),
            FileInfo(join(self.refdir,
                          'sequence_dataset/2017/001/sequence0002.txt'),
                     [datetime.datetime(2017, 1, 1, 12, 0),
                      datetime.datetime(2017, 1, 2, 0, 0)], {'id': 2}),

        ]
        assert found_files == check

        # Should find two files and should return them in two bins:
        found_files = list(
            datasets["sequence-placeholder"].find_files(
                "2017-01-01", "2017-01-02", bundle="6h",
            ))

        check = [
            [
                FileInfo(join(self.refdir,
                              'sequence_dataset/2017/001/sequence0001.txt'),
                         [datetime.datetime(2017, 1, 1, 0, 0),
                          datetime.datetime(2017, 1, 1, 12, 0)], {'id': 1}),
            ],
            [
                FileInfo(join(self.refdir,
                              'sequence_dataset/2017/001/sequence0002.txt'),
                         [datetime.datetime(2017, 1, 1, 12, 0),
                          datetime.datetime(2017, 1, 2, 0, 0)], {'id': 2}),
            ],
        ]
        assert found_files == check

    def _repr_files(self, files, comma=False):
        print("[")
        for file in files:
            if isinstance(file, FileInfo):
                print(self._repr_file_info(file))
            else:
                self._repr_files(file, True)

        if comma:
            print("],")
        else:
            print("]")

    def _repr_file_info(self, file_info):

        path = "join(self.refdir, '%s')" % (
            file_info.path[82:]
        )

        return "FileInfo({}, {}, {}),".format(
            path, repr(file_info.times), repr(file_info.attr)
        )
