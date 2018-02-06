import datetime
from os.path import dirname, join

import numpy as np
from typhon.spareice.datasets import Dataset, DatasetManager
from typhon.spareice.handlers import FileHandler, FileInfo, NetCDF4


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
                "tutorial_datasets/{satellite}/{year}/{month}/{day}/{hour}"
                "{minute}{second}-{end_hour}{end_minute}{end_second}.nc"
                ".{compression}"  # noqa
            ),
            name="tutorial",
            handler=NetCDF4(),
        )

        self.datasets += Dataset(
            join(self.refdir, "dataset_of_single_file.nc",),
            name="single",
            time_coverage=["2018-01-01", "2018-01-03"],
        )

        def sequence_get_info(file_info, **kwargs):
            """Small helper function for sequence dataset."""
            with open(file_info) as f:
                file_info.times[0] = datetime.datetime.strptime(
                    f.readline().rstrip(),
                    "Start: %Y-%m-%d %H:%M:%S"
                )
                file_info.times[1] = datetime.datetime.strptime(
                    f.readline().rstrip(),
                    "End: %Y-%m-%d %H:%M:%S"
                )
            return file_info

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
            placeholder={"id": "\d{4}"}
        )

        self.datasets += Dataset(
            join(self.refdir,
                 # NSS.HIRX.NJ.D99127.S0632.E0820.B2241718.WI.gz
                 "regex_dataset/NSS.HIR[XS].{satcode}.D{year2}{doy}.S{hour}"
                 "{minute}.E{end_hour}{end_minute}.B{B}.{station}.gz"
            ),
            name="regex-HIRS",
        )
        self.datasets["regex-HIRS"].set_placeholders(
            satcode=".{2}", B="\d{7}", station=".{2}",
        )

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
            ["2018-01-01", True],
            ["2018-01-01 06:00:00", True],
            [datetime.datetime(2018, 1, 1), True],
            [datetime.datetime(2018, 1, 1, 12,), True],
        ]

        for name, dataset in datasets.items():
            # print("Run test-contains for %s dataset:" % name)

            for timestamp, check in tests:
                # print("\tCheck coverage of %s (expected %s, got %s)" % (
                #     timestamp, check, timestamp in dataset
                # ))
                assert (timestamp in dataset) == check

    def test_glob(self):
        files = Dataset(
            join(
                self.refdir,
                "tutorial_datasets/{satellite}/*/*/*/*.nc.gz"
            ),
        )

        # Sort this after paths rather than times (because the times are all
        # equal)
        check = list(sorted([
            FileInfo(join(self.refdir,
                          'tutorial_datasets/SatelliteB/2018/01/02/180000-000000.nc.gz'),  # noqa
                     [datetime.datetime(1, 1, 1, 0, 0),
                      datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                     {'satellite': 'SatelliteB'}),
            FileInfo(join(self.refdir,
                          'tutorial_datasets/SatelliteB/2018/01/02/000000-060000.nc.gz'),  # noqa
                     [datetime.datetime(1, 1, 1, 0, 0),
                      datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                     {'satellite': 'SatelliteB'}),
            FileInfo(join(self.refdir,
                          'tutorial_datasets/SatelliteB/2018/01/02/120000-180000.nc.gz'),  # noqa
                     [datetime.datetime(1, 1, 1, 0, 0),
                      datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                     {'satellite': 'SatelliteB'}),
            FileInfo(join(self.refdir,
                          'tutorial_datasets/SatelliteB/2018/01/02/060000-120000.nc.gz'),  # noqa
                     [datetime.datetime(1, 1, 1, 0, 0),
                      datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                     {'satellite': 'SatelliteB'}),
            FileInfo(join(self.refdir,
                          'tutorial_datasets/SatelliteB/2018/01/01/180000-000000.nc.gz'),  # noqa
                     [datetime.datetime(1, 1, 1, 0, 0),
                      datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                     {'satellite': 'SatelliteB'}),
            FileInfo(join(self.refdir,
                          'tutorial_datasets/SatelliteB/2018/01/01/000000-060000.nc.gz'),  # noqa
                     [datetime.datetime(1, 1, 1, 0, 0),
                      datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                     {'satellite': 'SatelliteB'}),
            FileInfo(join(self.refdir,
                          'tutorial_datasets/SatelliteB/2018/01/01/120000-180000.nc.gz'),  # noqa
                     [datetime.datetime(1, 1, 1, 0, 0),
                      datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                     {'satellite': 'SatelliteB'}),
            FileInfo(join(self.refdir,
                          'tutorial_datasets/SatelliteB/2018/01/01/060000-120000.nc.gz'),  # noqa
                     [datetime.datetime(1, 1, 1, 0, 0),
                      datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                     {'satellite': 'SatelliteB'}),
            FileInfo(join(self.refdir,
                          'tutorial_datasets/SatelliteB/2018/01/03/000000-060000.nc.gz'),  # noqa
                     [datetime.datetime(1, 1, 1, 0, 0),
                      datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                     {'satellite': 'SatelliteB'}),
        ], key=lambda x: x.path))

        assert list(sorted(files, key=lambda x: x.path)) == check

    def test_magic_methods(self):
        """Test magic methods on the dataset examples of the tutorial.

        Returns:

        """
        datasets = self.init_datasets()
        filters = {"satellite": "SatelliteB"}
        result = datasets["tutorial"]["2018-01-01 03:00", filters]

        check = np.array(
          ['2018-01-01T00:00:00', '2018-01-01T00:09:13', '2018-01-01T00:18:27',
           '2018-01-01T00:27:41', '2018-01-01T00:36:55', '2018-01-01T00:46:09',
           '2018-01-01T00:55:23', '2018-01-01T01:04:36', '2018-01-01T01:13:50',
           '2018-01-01T01:23:04', '2018-01-01T01:32:18', '2018-01-01T01:41:32',
           '2018-01-01T01:50:46', '2018-01-01T01:59:59', '2018-01-01T02:09:13',
           '2018-01-01T02:18:27', '2018-01-01T02:27:41', '2018-01-01T02:36:55',
           '2018-01-01T02:46:09', '2018-01-01T02:55:23', '2018-01-01T03:04:36',
           '2018-01-01T03:13:50', '2018-01-01T03:23:04', '2018-01-01T03:32:18',
           '2018-01-01T03:41:32', '2018-01-01T03:50:46', '2018-01-01T03:59:59',
           '2018-01-01T04:09:13', '2018-01-01T04:18:27', '2018-01-01T04:27:41',
           '2018-01-01T04:36:55', '2018-01-01T04:46:09', '2018-01-01T04:55:23',
           '2018-01-01T05:04:36', '2018-01-01T05:13:50', '2018-01-01T05:23:04',
           '2018-01-01T05:32:18', '2018-01-01T05:41:32', '2018-01-01T05:50:46',
           '2018-01-01T05:59:59'], dtype="M8[s]")

        assert np.allclose(
            result["time"].astype("M8[s]").astype("int"),
            check.astype("int")
        )

    def test_tutorial(self):
        """Test the dataset examples of the tutorial.

        Returns:
            None
        """
        datasets = self.init_datasets()

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            datasets["tutorial"].find(
                "2017-12-31", "2018-01-01", no_files_error=False
            ))
        assert not empty

        # Find the closest file to 2018-01-01
        found_file = datasets["tutorial"].find_closest(
            "2018-01-01 03:00", filters={
                "!satellite": ("SatelliteA", "SatelliteC")
            }
        )

        # Limit this to SatelliteB
        refdir = join(self.refdir, "tutorial_datasets/SatelliteB")

        check = FileInfo(
            join(refdir, '2018/01/01/000000-060000.nc.gz'),
            [datetime.datetime(2018, 1, 1, 0, 0),
             datetime.datetime(2018, 1, 1, 6, 0)], {})

        assert found_file == check

        # Limit this dataset to SatelliteB permanently
        datasets["tutorial"].set_placeholders(
            satellite="SatelliteB",
        )

        # Should find four files:
        found_files = list(
            datasets["tutorial"].find(
                "2018-01-01", "2018-01-02",
            ))

        check = [
            FileInfo(join(refdir,
                          '2018/01/01/000000-060000.nc.gz'),
                     [datetime.datetime(2018, 1, 1, 0, 0),
                      datetime.datetime(2018, 1, 1, 6, 0)], {}),
            FileInfo(join(refdir,
                          '2018/01/01/060000-120000.nc.gz'),
                     [datetime.datetime(2018, 1, 1, 6, 0),
                      datetime.datetime(2018, 1, 1, 12, 0)], {}),
            FileInfo(join(refdir,
                          '2018/01/01/120000-180000.nc.gz'),
                     [datetime.datetime(2018, 1, 1, 12, 0),
                      datetime.datetime(2018, 1, 1, 18, 0)], {}),
            FileInfo(join(refdir,
                          '2018/01/01/180000-000000.nc.gz'),
                     [datetime.datetime(2018, 1, 1, 18, 0),
                      datetime.datetime(2018, 1, 2, 0, 0)], {}),

        ]

        assert found_files == check

        # Should find four files and should return them in two bins:
        found_files = list(
            datasets["tutorial"].find(
                "2018-01-01", "2018-01-02", bundle="12h",
            ))

        check = [
            [
                FileInfo(join(refdir,
                              '2018/01/01/000000-060000.nc.gz'),
                         [datetime.datetime(2018, 1, 1, 0, 0),
                          datetime.datetime(2018, 1, 1, 6, 0)], {}),
                FileInfo(join(refdir,
                              '2018/01/01/060000-120000.nc.gz'),
                         [datetime.datetime(2018, 1, 1, 6, 0),
                          datetime.datetime(2018, 1, 1, 12, 0)], {}),
            ],
            [
                FileInfo(join(refdir,
                              '2018/01/01/120000-180000.nc.gz'),
                         [datetime.datetime(2018, 1, 1, 12, 0),
                          datetime.datetime(2018, 1, 1, 18, 0)], {}),
                FileInfo(join(refdir,
                              '2018/01/01/180000-000000.nc.gz'),
                         [datetime.datetime(2018, 1, 1, 18, 0),
                          datetime.datetime(2018, 1, 2, 0, 0)], {}),
            ],
        ]

        assert found_files == check

        # Should find four files and should return them in two bins:
        found_files = list(
            datasets["tutorial"].find(
                "2018-01-01", "2018-01-02", bundle=3,
            ))

        check = [
            [
                FileInfo(join(refdir,
                              '2018/01/01/000000-060000.nc.gz'),
                         [datetime.datetime(2018, 1, 1, 0, 0),
                          datetime.datetime(2018, 1, 1, 6, 0)], {}),
                FileInfo(join(refdir,
                              '2018/01/01/060000-120000.nc.gz'),
                         [datetime.datetime(2018, 1, 1, 6, 0),
                          datetime.datetime(2018, 1, 1, 12, 0)], {}),
                FileInfo(join(refdir,
                              '2018/01/01/120000-180000.nc.gz'),
                         [datetime.datetime(2018, 1, 1, 12, 0),
                          datetime.datetime(2018, 1, 1, 18, 0)], {}),
            ],
            [
                FileInfo(join(refdir,
                              '2018/01/01/180000-000000.nc.gz'),
                         [datetime.datetime(2018, 1, 1, 18, 0),
                          datetime.datetime(2018, 1, 2, 0, 0)], {}),
            ],
        ]

        assert found_files == check

        for test_method in [Dataset.map, Dataset.imap]:
            # Check map method
            _, results = zip(*test_method(
                datasets["tutorial"],
                "2018-01-01", "2018-01-03", func=TestDataset._tutorial_map
            ))
            check = ('gz', 'gz', 'gz', 'gz', 'gz', 'gz', 'gz', 'gz')
            assert results == check

            # Check map method on content
            _, results = zip(*test_method(
                datasets["tutorial"],
                "2018-01-01", "2018-01-03",
                func=TestDataset._tutorial_map_content,on_content=True,
            ))
            check = (
                0.25007269785924874, 0.25007269785924874, 0.25007269785924874,
                0.25007269785924874, 0.25007269785924874, 0.25007269785924874,
                0.25007269785924874, 0.25007269785924874)
            assert np.allclose(results, check)

    @staticmethod
    def _tutorial_map(file_info):
        return file_info.attr["compression"]

    @staticmethod
    def _tutorial_map_content(data, file_info):
        return data["data"].mean().item(0)

    def test_files_overlap_subdirectory(self):
        """A file covers a time period longer than its sub directory.
        """
        datasets = self.init_datasets()
        datasets["tutorial"].set_placeholders(
            satellite="SatelliteA"
        )
        found_file = datasets["tutorial"].find_closest("2018-01-03")

        check = FileInfo(
            join(self.refdir,
                 'tutorial_datasets/SatelliteA/2018/01/02/210000-020000.nc.zip'
                 ),
            [datetime.datetime(2018, 1, 2, 21, 0),
             datetime.datetime(2018, 1, 3, 2, 0)],
             {'satellite': 'SatelliteA', 'compression': 'zip'}
        )

        assert found_file == check

    def test_single(self):
        """Test find on the single dataset.

        Returns:
            None
        """
        datasets = self.init_datasets()

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            datasets["single"].find(
                "2016-12-31", "2018-01-01", no_files_error=False
            ))
        assert not empty

        check = [
            FileInfo(join(self.refdir, 'dataset_of_single_file.nc'),
                     [datetime.datetime(2018, 1, 1, 0, 0),
                      datetime.datetime(2018, 1, 3, 0, 0)], {}),
        ]

        found_files = list(
            datasets["single"].find(
                "2018-01-01", "2018-01-02",
            ))

        assert found_files == check

        found_files = list(
            datasets["single"].find(
                "2018-01-01", "2018-01-02", bundle="12h",
            ))

        assert found_files == check

        found_files = list(
            datasets["single"].find(
                "2018-01-01", "2018-01-02", bundle=3,
            ))

        assert found_files == check

    def test_sequence(self):
        """Test find on the sequence datasets.

        Returns:
            None
        """
        datasets = self.init_datasets()

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            datasets["sequence-placeholder"].find(
                "2016-12-31", "2018-01-01", no_files_error=False
            ))
        assert not empty

        # Should find two files:
        found_files = list(
            datasets["sequence-placeholder"].find(
                "2018-01-01", "2018-01-02",
            ))

        check = [
            FileInfo(join(self.refdir,
                          'sequence_dataset/2018/001/sequence0001.txt'),
                     [datetime.datetime(2018, 1, 1, 0, 0),
                      datetime.datetime(2018, 1, 1, 12, 0)], {'id': 1}),
            FileInfo(join(self.refdir,
                          'sequence_dataset/2018/001/sequence0002.txt'),
                     [datetime.datetime(2018, 1, 1, 12, 0),
                      datetime.datetime(2018, 1, 2, 0, 0)], {'id': 2}),

        ]
        assert found_files == check

        # Should find two files and should return them in two bins:
        found_files = list(
            datasets["sequence-placeholder"].find(
                "2018-01-01", "2018-01-02", bundle="6h",
            ))

        check = [
            [
                FileInfo(join(self.refdir,
                              'sequence_dataset/2018/001/sequence0001.txt'),
                         [datetime.datetime(2018, 1, 1, 0, 0),
                          datetime.datetime(2018, 1, 1, 12, 0)], {'id': 1}),
            ],
            [
                FileInfo(join(self.refdir,
                              'sequence_dataset/2018/001/sequence0002.txt'),
                         [datetime.datetime(2018, 1, 1, 12, 0),
                          datetime.datetime(2018, 1, 2, 0, 0)], {'id': 2}),
            ],
        ]
        assert found_files == check

    def test_sequence_placeholder(self):
        """Test find on all standard datasets.

        Returns:
            None
        """
        datasets = self.init_datasets()

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            datasets["sequence-placeholder"].find(
                "2016-12-31", "2018-01-01", no_files_error=False
            ))
        assert not empty

        # Should find two files:
        found_files = list(
            datasets["sequence-placeholder"].find(
                "2018-01-01", "2018-01-02",
            ))

        check = [
            FileInfo(join(self.refdir,
                          'sequence_dataset/2018/001/sequence0001.txt'),
                     [datetime.datetime(2018, 1, 1, 0, 0),
                      datetime.datetime(2018, 1, 1, 12, 0)], {'id': 1}),
            FileInfo(join(self.refdir,
                          'sequence_dataset/2018/001/sequence0002.txt'),
                     [datetime.datetime(2018, 1, 1, 12, 0),
                      datetime.datetime(2018, 1, 2, 0, 0)], {'id': 2}),

        ]
        assert found_files == check

        # Should find two files and should return them in two bins:
        found_files = list(
            datasets["sequence-placeholder"].find(
                "2018-01-01", "2018-01-02", bundle="6h",
            ))

        check = [
            [
                FileInfo(join(self.refdir,
                              'sequence_dataset/2018/001/sequence0001.txt'),
                         [datetime.datetime(2018, 1, 1, 0, 0),
                          datetime.datetime(2018, 1, 1, 12, 0)], {'id': 1}),
            ],
            [
                FileInfo(join(self.refdir,
                              'sequence_dataset/2018/001/sequence0002.txt'),
                         [datetime.datetime(2018, 1, 1, 12, 0),
                          datetime.datetime(2018, 1, 2, 0, 0)], {'id': 2}),
            ],
        ]
        assert found_files == check

    def test_regex(self):
        datasets = self.init_datasets()

        check = [
            FileInfo(join(self.refdir,
                          'regex_dataset/NSS.HIRX.NJ.D99127.S0632.E0820.B2241718.WI.gz'),  # noqa
                     [datetime.datetime(1999, 5, 7, 6, 32),
                      datetime.datetime(1999, 5, 7, 8, 20)],
                     {'satcode': 'NJ', 'B': '2241718',
                      'station': 'WI'}),
        ]

        found_file = datasets["regex-HIRS"].find_closest("1999-05-08")

        assert found_file == check[0]
        assert found_file.attr == check[0].attr

        found_files = \
            list(datasets["regex-HIRS"].find("1999-05-07", "1999-05-09"))

        assert found_files == check

    def _print_files(self, files, comma=False):
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

        return "FileInfo(\n\t{}, \t{}, \t{}),".format(
            path, repr(file_info.times), repr(file_info.attr)
        )
