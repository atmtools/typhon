import os
from posixpath import join  # LocalFileSystem always uses /

import datetime
import numpy as np
import pytest
import logging

from typhon.files import FileHandler, FileInfo, FileSet, FileSetManager
from typhon.files.utils import get_testfiles_directory


class TestFileSet:
    """Testing the fileset methods."""

    filesets = None
    # replace separator, fsspec always uses forward /
    refdir = get_testfiles_directory("filesets").replace(os.sep, "/")

    @pytest.fixture
    def file_system(self, request, tmp_path):
        """Prepare filesystem.

        Prepare a filesystem to test if the filesystem spec works with it.
        """
        if request.param == "zip":
            from fsspec.implementations.zip import ZipFileSystem
            from shutil import make_archive
            # prepare archive to test on
            archive = tmp_path / "test"
            make_archive(
                archive, "zip", "/" if os.path.isabs(self.refdir) else os.curdir,
                self.refdir)

            return ZipFileSystem(archive.with_suffix(".zip"))
        elif request.param == "local":
            from fsspec.implementations.local import LocalFileSystem
            return LocalFileSystem()

    def _refdir_for_fs(self, fs):
        """Get appropriate reference dir for fs.

        Some fs, such as ZipFileSystem or S3FS, don't return leading /, then
        our refdir shouldn't contain this either.
        """

        # check if this filesystem returns absolute or relative and adapt
        # refdir accordingly
        try:
            cont = fs.ls(self.refdir)
            # on windows, this appears empty (?!) if we have a zip and refdir
            # contains a drive
            if len(cont) == 0 and self.refdir[1] == ":":
                return self.refdir[3:]
            fn = cont[0]
            if fn[0] not in (os.sep, "/", self.refdir[0]):
                return self.refdir[1:]
        except (IndexError, AttributeError, StopIteration, KeyError, TypeError):
            pass
        return self.refdir

    def init_filesets(self, fs=None):
        self.filesets = FileSetManager()

        refdir = self._refdir_for_fs(fs)
        self.filesets += FileSet(
            join(
                refdir,
                "tutorial", "{satellite}", "{year}-{month}-{day}",
                "{hour}{minute}{second}-{end_hour}{end_minute}{end_second}.nc"
            ),
            name="tutorial",
            fs=fs
        )

        self.filesets += FileSet(
            join(refdir, "single_file.nc",),
            name="single",
            time_coverage=["2018-01-01", "2018-01-03"],
            fs=fs
        )

        def sequence_get_info(file_info, **kwargs):
            """Small helper function for sequence fileset."""
            with file_info.file_system.open(file_info, mode="rt") as f:
                file_info.times[0] = datetime.datetime.strptime(
                    f.readline().rstrip(),
                    "Start: %Y-%m-%d %H:%M:%S"
                )
                file_info.times[1] = datetime.datetime.strptime(
                    f.readline().rstrip(),
                    "End: %Y-%m-%d %H:%M:%S"
                )
            return file_info

        self.filesets += FileSet(
            join(refdir, "sequence",
                 "{year}", "{doy}", "sequence*.txt",),
            name="sequence-wildcard",
            handler=FileHandler(
                info=sequence_get_info,
            ),
            info_via="handler",
            fs=fs
        )
        self.filesets += FileSet(
            join(refdir, "sequence",
                 "{year}", "{doy}", "sequence{id}.txt",
            ),
            handler=FileHandler(
                info=sequence_get_info,
            ),
            name="sequence-placeholder",
            info_via="both",
            placeholder={"id": r"\d{4}"},
            fs=fs
        )

        self.filesets += FileSet(
            join(refdir,
                 # NSS.HIRX.NJ.D99127.S0632.E0820.B2241718.WI.gz
                 "regex", "NSS.HIR[XS].{satcode}.D{year2}{doy}.S{hour}"
                 "{minute}.E{end_hour}{end_minute}.B{B}.{station}.gz"
            ),
            name="regex-HIRS",
            fs=fs
        )
        self.filesets["regex-HIRS"].set_placeholders(
            satcode=".{2}", B=r"\d{7}", station=".{2}",
        )

        return self.filesets

    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    @pytest.mark.parametrize("file_system", [None, "local", "zip"], indirect=True)
    def test_contains(self, file_system):
        """Test whether all filesets cover the testing timestamps.

        Returns:
            None
        """
        filesets = self.init_filesets(file_system)
        tests = [
            # [Timestamp(s), Should it be covered by the filesets?]
            ["2016-01-01", False],
            ["2018-01-01", True],
            ["2018-01-01 06:00:00", True],
            [datetime.datetime(2018, 1, 1), True],
            [datetime.datetime(2018, 1, 1, 12,), True],
        ]

        for name, fileset in filesets.items():
            # print("Run test-contains for %s fileset:" % name)

            for timestamp, check in tests:
                # print("\tCheck coverage of %s (expected %s, got %s)" % (
                #     timestamp, check, timestamp in fileset
                # ))
                assert (timestamp in fileset) == check

    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    @pytest.mark.parametrize("file_system", [None, "local", "zip"], indirect=True)
    def test_glob(self, file_system):
        refdir = self._refdir_for_fs(file_system)
        files = FileSet(
            join(
                refdir,
                "tutorial", "{satellite}", "*", "*.nc"
            ),
            placeholder={"satellite": 'SatelliteA'},
            fs=file_system
        )

        # Sort this after paths rather than times (because the times are all
        # equal)
        check = list(sorted([
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-02', '000000-040000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-02', '080000-120000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-02', '200000-000000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-02', '040000-080000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-02', '120000-160000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-02', '160000-200000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-01', '000000-040000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-01', '080000-120000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-01', '200000-000000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-01', '040000-080000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-01', '120000-160000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteA', '2018-01-01', '160000-200000.nc'),
                [datetime.datetime(1, 1, 1, 0, 0),
                 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)],
                {'satellite': 'SatelliteA'}),
        ], key=lambda x: x.path))

        assert list(sorted(files, key=lambda x: x.path)) == check

    @pytest.mark.skip
    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    def test_magic_methods(self, file_system):
        """Test magic methods on the fileset examples of the tutorial.

        Returns:

        """
        filesets = self.init_filesets(file_system)
        filters = {"satellite": "SatelliteB"}
        result = filesets["tutorial"]["2018-01-01 03:00", filters]

        check = np.array(
            ['2018-01-01T00:00:00.000000000', '2018-01-01T00:20:00.000000000',
             '2018-01-01T00:40:00.000000000', '2018-01-01T01:00:00.000000000',
             '2018-01-01T01:20:00.000000000', '2018-01-01T01:40:00.000000000',
             '2018-01-01T02:00:00.000000000', '2018-01-01T02:20:00.000000000',
             '2018-01-01T02:40:00.000000000', '2018-01-01T03:00:00.000000000',
             '2018-01-01T03:20:00.000000000', '2018-01-01T03:40:00.000000000',
             '2018-01-01T04:00:00.000000000', '2018-01-01T04:20:00.000000000',
             '2018-01-01T04:40:00.000000000'], dtype='datetime64[ns]')

        assert np.allclose(
            result["time"].astype("int"),
            check.astype("int")
        )

    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    @pytest.mark.parametrize("file_system", [None, "local"], indirect=True)
    def test_tutorial(self, file_system):
        """Test the fileset examples of the tutorial.

        Returns:
            None
        """
        filesets = self.init_filesets(file_system)
        refdir = self._refdir_for_fs(file_system)

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            filesets["tutorial"].find(
                "2017-12-31", "2018-01-01", no_files_error=False
            ))
        assert not empty

        # Find the closest file to 2018-01-01, limited to SatelliteB
        # temporarily:
        found_file = filesets["tutorial"].find_closest(
            "2018-01-01 03:00", filters={
                "!satellite": ("SatelliteA", "SatelliteC")
            }
        )

        #print("closest check", self._repr_file_info(found_file))

        check = FileInfo(
            join(refdir, 'tutorial',
                 'SatelliteB', '2018-01-01', '000000-050000.nc'),
            [datetime.datetime(2018, 1, 1, 0, 0),
             datetime.datetime(2018, 1, 1, 5, 0)], {'satellite': 'SatelliteB'})

        assert found_file == check

        # Limit this fileset to SatelliteB permanently
        filesets["tutorial"].set_placeholders(
            satellite="SatelliteB",
        )

        # Should find four files:
        found_files = list(
            filesets["tutorial"].find(
                "2018-01-01", "2018-01-02",
            ))

        #print("four files:")
        # self._print_files(found_files)

        check = [
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteB', '2018-01-01', '000000-050000.nc'),
                [datetime.datetime(2018, 1, 1, 0, 0),
                 datetime.datetime(2018, 1, 1, 5, 0)],
                {'satellite': 'SatelliteB'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteB', '2018-01-01', '050000-100000.nc'),
                [datetime.datetime(2018, 1, 1, 5, 0),
                 datetime.datetime(2018, 1, 1, 10, 0)],
                {'satellite': 'SatelliteB'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteB', '2018-01-01', '100000-150000.nc'),
                [datetime.datetime(2018, 1, 1, 10, 0),
                 datetime.datetime(2018, 1, 1, 15, 0)],
                {'satellite': 'SatelliteB'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteB', '2018-01-01', '150000-200000.nc'),
                [datetime.datetime(2018, 1, 1, 15, 0),
                 datetime.datetime(2018, 1, 1, 20, 0)],
                {'satellite': 'SatelliteB'}),
            FileInfo(
                join(refdir, 'tutorial',
                     'SatelliteB', '2018-01-01', '200000-010000.nc'),
                [datetime.datetime(2018, 1, 1, 20, 0),
                 datetime.datetime(2018, 1, 2, 1, 0)],
                {'satellite': 'SatelliteB'}),
        ]

        assert found_files == check

        # Should find four files and should return them in two bins:
        found_files = list(
            filesets["tutorial"].find(
                "2018-01-01", "2018-01-02", bundle="12h",
            ))

        # print("Bundle 12h:")
        # self._print_files(found_files)

        check = [
            [
                FileInfo(
                    join(refdir, 'tutorial',
                         'SatelliteB', '2018-01-01', '000000-050000.nc'),
                    [datetime.datetime(2018, 1, 1, 0, 0),
                     datetime.datetime(2018, 1, 1, 5, 0)],
                    {'satellite': 'SatelliteB'}),
                FileInfo(
                    join(refdir, 'tutorial',
                         'SatelliteB', '2018-01-01', '050000-100000.nc'),
                    [datetime.datetime(2018, 1, 1, 5, 0),
                     datetime.datetime(2018, 1, 1, 10, 0)],
                    {'satellite': 'SatelliteB'}),
                FileInfo(
                    join(refdir, 'tutorial',
                         'SatelliteB', '2018-01-01', '100000-150000.nc'),
                    [datetime.datetime(2018, 1, 1, 10, 0),
                     datetime.datetime(2018, 1, 1, 15, 0)],
                    {'satellite': 'SatelliteB'}),
            ],
            [
                FileInfo(
                    join(refdir, 'tutorial',
                         'SatelliteB', '2018-01-01', '150000-200000.nc'),
                    [datetime.datetime(2018, 1, 1, 15, 0),
                     datetime.datetime(2018, 1, 1, 20, 0)],
                    {'satellite': 'SatelliteB'}),
                FileInfo(
                    join(refdir, 'tutorial',
                         'SatelliteB', '2018-01-01', '200000-010000.nc'),
                    [datetime.datetime(2018, 1, 1, 20, 0),
                     datetime.datetime(2018, 1, 2, 1, 0)],
                    {'satellite': 'SatelliteB'}),
            ],
        ]

        assert found_files == check

        # Should find four files and should return them in two bins:
        found_files = list(
            filesets["tutorial"].find(
                "2018-01-01", "2018-01-02", bundle=3,
            ))

        # print("Bundle 3:")
        # self._print_files(found_files)

        check = [
            [
                FileInfo(
                    join(refdir, 'tutorial',
                         'SatelliteB', '2018-01-01', '000000-050000.nc'),
                    [datetime.datetime(2018, 1, 1, 0, 0),
                     datetime.datetime(2018, 1, 1, 5, 0)],
                    {'satellite': 'SatelliteB'}),
                FileInfo(
                    join(refdir, 'tutorial',
                         'SatelliteB', '2018-01-01', '050000-100000.nc'),
                    [datetime.datetime(2018, 1, 1, 5, 0),
                     datetime.datetime(2018, 1, 1, 10, 0)],
                    {'satellite': 'SatelliteB'}),
                FileInfo(
                    join(refdir, 'tutorial',
                         'SatelliteB', '2018-01-01', '100000-150000.nc'),
                    [datetime.datetime(2018, 1, 1, 10, 0),
                     datetime.datetime(2018, 1, 1, 15, 0)],
                    {'satellite': 'SatelliteB'}),
            ],
            [
                FileInfo(
                    join(refdir, 'tutorial',
                         'SatelliteB', '2018-01-01', '150000-200000.nc'),
                    [datetime.datetime(2018, 1, 1, 15, 0),
                     datetime.datetime(2018, 1, 1, 20, 0)],
                    {'satellite': 'SatelliteB'}),
                FileInfo(
                    join(refdir, 'tutorial',
                         'SatelliteB', '2018-01-01', '200000-010000.nc'),
                    [datetime.datetime(2018, 1, 1, 20, 0),
                     datetime.datetime(2018, 1, 2, 1, 0)],
                    {'satellite': 'SatelliteB'}),
            ],
        ]

        assert found_files == check

        for test_method in [FileSet.map, FileSet.imap]:
            # Check map method
            results = list(test_method(
                filesets["tutorial"], TestFileSet._tutorial_map,
                start="2018-01-01", end="2018-01-03"
            ))
            check = ['SatelliteB', 'SatelliteB', 'SatelliteB', 'SatelliteB',
                     'SatelliteB', 'SatelliteB', 'SatelliteB', 'SatelliteB',
                     'SatelliteB', 'SatelliteB']
            assert results == check

            # Check map method on content
            results = list(test_method(
                filesets["tutorial"], TestFileSet._tutorial_map_content,
                start="2018-01-01", end="2018-01-03", on_content=True,
            ))
            check = [111.92121062601221, 24.438060320121387,
                     -98.80775640366036, -75.84330354813459, 59.41297628327247,
                     106.80513550614192, -3.999061608822918,
                     -108.68523313569861, -51.82441769876156, 66.33842832792985
            ]
            assert np.allclose(results, check)

    @staticmethod
    def _tutorial_map(file_info):
        return file_info.attr["satellite"]

    @staticmethod
    def _tutorial_map_content(data,):
        return data["data"].mean().item(0)

    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    @pytest.mark.parametrize("file_system", [None, "local", "zip"], indirect=True)
    def test_files_overlap_subdirectory(self, file_system):
        """A file covers a time period longer than its sub directory.
        """
        filesets = self.init_filesets(file_system)
        filesets["tutorial"].set_placeholders(
            satellite="SatelliteA"
        )
        found_file = filesets["tutorial"].find_closest("2018-01-03")

        check = FileInfo(
            join(self._refdir_for_fs(file_system), 'tutorial',
                 'SatelliteA', '2018-01-02', '200000-000000.nc'),
            [datetime.datetime(2018, 1, 2, 20, 0),
             datetime.datetime(2018, 1, 3, 0, 0)],
            {'satellite': 'SatelliteA'}
        )

        assert found_file == check

    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    @pytest.mark.parametrize("file_system", [None, "local", "zip"], indirect=True)
    def test_single(self, file_system):
        """Test find on the single fileset.

        Returns:
            None
        """
        filesets = self.init_filesets(file_system)

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            filesets["single"].find(
                "2016-12-31", "2018-01-01", no_files_error=False
            ))
        assert not empty

        check = [
            FileInfo(join(self._refdir_for_fs(file_system), 'single_file.nc'),
                     [datetime.datetime(2018, 1, 1, 0, 0),
                      datetime.datetime(2018, 1, 3, 0, 0)], {}),
        ]

        found_files = list(
            filesets["single"].find(
                "2018-01-01", "2018-01-02",
            ))

        assert found_files == check

        found_files = list(
            filesets["single"].find(
                "2018-01-01", "2018-01-02", bundle="12h",
            ))

        assert found_files == check

        found_files = list(
            filesets["single"].find(
                "2018-01-01", "2018-01-02", bundle=3,
            ))

        assert found_files == check

    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    @pytest.mark.parametrize("file_system", [None, "local", "zip"], indirect=True)
    def test_sequence(self, file_system):
        """Test find on the sequence filesets.

        Returns:
            None
        """
        filesets = self.init_filesets(file_system)
        refdir = self._refdir_for_fs(file_system)

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            filesets["sequence-placeholder"].find(
                "2016-12-31", "2018-01-01", no_files_error=False
            ))
        assert not empty

        # Should find two files:
        found_files = list(
            filesets["sequence-placeholder"].find(
                "2018-01-01", "2018-01-02",
            ))

        check = [
            FileInfo(join(refdir, 'sequence',
                          '2018', '001', 'sequence0001.txt'),
                     [datetime.datetime(2018, 1, 1, 0, 0),
                      datetime.datetime(2018, 1, 1, 12, 0)], {'id': 1}),
            FileInfo(join(refdir, 'sequence',
                          '2018', '001', 'sequence0002.txt'),
                     [datetime.datetime(2018, 1, 1, 12, 0),
                      datetime.datetime(2018, 1, 2, 0, 0)], {'id': 2}),

        ]
        assert found_files == check

        # Should find two files and should return them in two bins:
        found_files = list(
            filesets["sequence-placeholder"].find(
                "2018-01-01", "2018-01-02", bundle="6h",
            ))

        check = [
            [
                FileInfo(join(refdir, 'sequence',
                              '2018', '001', 'sequence0001.txt'),
                         [datetime.datetime(2018, 1, 1, 0, 0),
                          datetime.datetime(2018, 1, 1, 12, 0)], {'id': 1}),
            ],
            [
                FileInfo(join(refdir, 'sequence',
                              '2018', '001', 'sequence0002.txt'),
                         [datetime.datetime(2018, 1, 1, 12, 0),
                          datetime.datetime(2018, 1, 2, 0, 0)], {'id': 2}),
            ],
        ]
        assert found_files == check


    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    @pytest.mark.parametrize("file_system", [None, "local", "zip"], indirect=True)
    def test_logs(self, file_system, caplog):
        """Test that searching logs as expected."""
        filesets = self.init_filesets(file_system)
        refdir = self._refdir_for_fs(file_system)
        # should log
        with caplog.at_level(logging.DEBUG):
            found_files = list(
                filesets["sequence-placeholder"].find(
                    "2018-01-01", "2018-01-02",
                ))
        assert ("Find files for sequence-placeholder "
                "between 2018-01-01 00:00:00 and "
                "2018-01-01 23:59:59") in caplog.text
        assert f"via file system" in caplog.text


    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    @pytest.mark.parametrize("file_system", [None, "local", "zip"], indirect=True)
    def test_sequence_placeholder(self, file_system):
        """Test find on all standard filesets.

        Returns:
            None
        """
        filesets = self.init_filesets(file_system)
        refdir = self._refdir_for_fs(file_system)

        # STANDARD DATASET
        # Should not find anything:
        empty = list(
            filesets["sequence-placeholder"].find(
                "2016-12-31", "2018-01-01", no_files_error=False
            ))
        assert not empty

        # Should find two files:
        found_files = list(
            filesets["sequence-placeholder"].find(
                "2018-01-01", "2018-01-02",
            ))

        check = [
            FileInfo(join(refdir, 'sequence',
                          '2018', '001', 'sequence0001.txt'),
                     [datetime.datetime(2018, 1, 1, 0, 0),
                      datetime.datetime(2018, 1, 1, 12, 0)], {'id': 1}),
            FileInfo(join(refdir, 'sequence',
                          '2018', '001', 'sequence0002.txt'),
                     [datetime.datetime(2018, 1, 1, 12, 0),
                      datetime.datetime(2018, 1, 2, 0, 0)], {'id': 2}),

        ]
        assert found_files == check

        # Should find two files and should return them in two bins:
        found_files = list(
            filesets["sequence-placeholder"].find(
                "2018-01-01", "2018-01-02", bundle="6h",
            ))

        check = [
            [
                FileInfo(join(refdir, 'sequence',
                              '2018', '001', 'sequence0001.txt'),
                         [datetime.datetime(2018, 1, 1, 0, 0),
                          datetime.datetime(2018, 1, 1, 12, 0)], {'id': 1}),
            ],
            [
                FileInfo(join(refdir, 'sequence',
                              '2018', '001', 'sequence0002.txt'),
                         [datetime.datetime(2018, 1, 1, 12, 0),
                          datetime.datetime(2018, 1, 2, 0, 0)], {'id': 2}),
            ],
        ]
        assert found_files == check

    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    @pytest.mark.parametrize("file_system", [None, "local", "zip"], indirect=True)
    def test_regex(self, file_system):
        filesets = self.init_filesets(file_system)

        check = [
            FileInfo(join(self._refdir_for_fs(file_system), 'regex',
                          'NSS.HIRX.NJ.D99127.S0632.E0820.B2241718.WI.gz'),
                     [datetime.datetime(1999, 5, 7, 6, 32),
                      datetime.datetime(1999, 5, 7, 8, 20)],
                     {'satcode': 'NJ', 'B': '2241718',
                      'station': 'WI'}),
        ]

        found_file = filesets["regex-HIRS"].find_closest("1999-05-08")

        assert found_file == check[0]
        assert found_file.attr == check[0].attr

        found_files = \
            list(filesets["regex-HIRS"].find("1999-05-07", "1999-05-09"))

        assert found_files == check

    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    @pytest.mark.parametrize("file_system", [None, "local", "zip"], indirect=True)
    def test_complicated_subdirs(self, file_system):
        """Check whether FileSet can find files in subdirectories that contain
        text and placeholders.
        """
        refdir = self._refdir_for_fs(file_system)
        # The Pinocchio fileset from the cloud toolbox: a folder name contains
        # normal text and a placeholder:
        pinocchio = FileSet(
            join(refdir, "pinocchio",
                 "t{year2}{month}{day}",
                 "tm{year2}{month}{day}{hour}{minute}{second}{millisecond}.jpg",
                 ),
            fs=file_system
        )

        # Find all files:
        files = list(pinocchio)

        check = [
            FileInfo(
                join(refdir,
                     'pinocchio', 't171102', 'tm171102132855573.jpg'),
                [datetime.datetime(2017, 11, 2, 13, 28, 55, 573000),
                 datetime.datetime(2017, 11, 2, 13, 28, 55, 573000)], {}),
        ]
        assert files == check

    @pytest.mark.skip
    def test_align(self):
        """Test the align method.
        """
        # Find an adequate test

        filesets = self.init_filesets()

        start, end = "2018-01-01", "2018-01-02"
        a_dataset = filesets["tutorial"].copy()
        a_dataset.name = "SatelliteA"
        a_dataset.set_placeholders(satellite="SatelliteA")
        a_reference = a_dataset.collect(start, end)
        b_dataset = filesets["tutorial"].copy()
        b_dataset.name = "SatelliteB"
        b_dataset.set_placeholders(satellite="SatelliteB")
        b_reference = b_dataset.collect(start, end)

        all_data = {}
        for files, data in a_dataset.align(start, end, [b_dataset]):
            for dataset in data:
                if dataset in all_data:
                    all_data[dataset] = GroupedArrays.concat(
                        [all_data[dataset], *data[dataset]]
                    )
                else:
                    all_data[dataset] = GroupedArrays.concat(
                        data[dataset]
                    )

        a_retrieved = all_data["SatelliteA"]
        b_retrieved = all_data["SatelliteB"]

        assert a_reference == a_retrieved
        assert b_reference == b_retrieved

    def _repr_file_info(self, file_info):

        path = "join(self.refdir, '%s')" % (
            file_info.path[len(self.refdir)+1:]
        )

        return "FileInfo(\n\t{}, \t{}, \t{}),".format(
            path, repr(file_info.times), repr(file_info.attr)
        )

    def test_compare_fileinfo(self):
        """Test comparing two FileInfo instances."""
        f1 = FileInfo(
                path="fake/path",
                times=[datetime.datetime(1900, 1, 1, 0),
                       datetime.datetime(1900, 1, 1, 2)],
                attr={})
        f2 = FileInfo(
                path="fake/path",
                times=[datetime.datetime(1900, 1, 1, 0),
                       datetime.datetime(1900, 1, 1, 2)],
                attr={})
        f3 = FileInfo(
                path="other/fake/path",
                times=[datetime.datetime(1900, 1, 1, 0),
                       datetime.datetime(1900, 1, 1, 2)],
                attr={})
        f4 = FileInfo(
                path="fake/path",
                times=[datetime.datetime(1910, 1, 1, 0),
                       datetime.datetime(1910, 1, 1, 2)],
                attr={})
        assert f1 == f2
        assert f1 != f3
        assert f1 != f4
        assert f2 != f3
        assert f2 != f4
        assert f3 != f4
        assert f1 != "fake/path"
