from os.path import dirname, join
import sys
import warnings
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from typhon.collocations import collapse, Collocator, Collocations, expand
from typhon.files import FileSet, MHS_HDF, NetCDF4
from typhon.files.utils import get_testfiles_directory
import xarray as xr


class TestCollocations:
    """Testing the collocation functions."""

    refdir = get_testfiles_directory("collocations")

    @pytest.mark.skipif(refdir is None, reason="typhon-testfiles not found.")
    def test_search(self):
        """Collocate fake MHS filesets"""
        fake_mhs1 = FileSet(
            path=join(self.refdir,
                "{satname}_mhs_{year}", "{month}", "{day}", 
                "*NSS.MHSX.*.S{hour}{minute}.E{end_hour}{end_minute}.*.h5"),
            handler=MHS_HDF(),
        )
        fake_mhs2 = fake_mhs1.copy()

        with TemporaryDirectory() as outdir:
            collocations = Collocations(
                path=join(outdir, "{year}-{month}-{day}", 
                                  "{hour}{minute}{second}-{end_hour}{end_minute}{end_second}"),
            )
            collocations.search(
                [fake_mhs1, fake_mhs2],
                start="2007",
                end="2008", 
                max_interval="1h",
                max_distance="10km"
            )

    def test_flat_to_main_coord(self):
        """Tests Collocator._flat_to_main_coord

        This method is crucial since it stacks the whole input datasets for the
        collocating routine and makes them collocateable.
        """
        collocator = Collocator()

        test = xr.Dataset({
            "time": ("time", np.arange(10)),
            "lat": ("time", np.arange(10)),
            "lon": ("time", np.arange(10)),
        })
        check = xr.Dataset({
            "time": ("collocation", np.arange(10)),
            "lat": ("collocation", np.arange(10)),
            "lon": ("collocation", np.arange(10)),
        })
        results = collocator._flat_to_main_coord(test)
        assert check.equals(results)

        test = xr.Dataset({
            "time": ("main", np.arange(10)),
            "lat": ("main", np.arange(10)),
            "lon": ("main", np.arange(10)),
        })
        check = xr.Dataset({
            "time": ("collocation", np.arange(10)),
            "lat": ("collocation", np.arange(10)),
            "lon": ("collocation", np.arange(10)),
        })
        results = collocator._flat_to_main_coord(test)
        assert check.equals(results)

        test = xr.Dataset({
            "time": ("scnline", np.arange(5)),
            "lat": (("scnline", "scnpos"), np.arange(10).reshape(5, 2)),
            "lon": (("scnline", "scnpos"), np.arange(10).reshape(5, 2)),
        })
        check = test.stack(collocation=("scnline", "scnpos"))
        results = collocator._flat_to_main_coord(test)
        assert check.equals(results)

    def test_collocate_collapse_expand(self):
        """Test whether collocating, collapsing and expanding work"""
        collocator = Collocator()

        test = xr.Dataset({
            "time": ("time", np.arange("2000", "2010", dtype="M8[Y]").astype('datetime64[ns]')),
            "lat": ("time", np.arange(10)),
            "lon": ("time", np.arange(10)),
        })

        collocations = collocator.collocate(
            test, test, max_interval="30 days",
            max_distance="150 miles"
        )

        collapsed = collapse(collocations)
        expanded = expand(collocations)

    def test_collocate_matches_skips_unreadable_files(self):
        """_collocate_matches skips matches whose files could not be read.

        When align yields None (because a file could not be read and
        skip_file_errors is True), the affected match must be skipped
        instead of crashing with an AttributeError on the missing data.
        """
        collocator = Collocator()

        def make_ds():
            return xr.Dataset({
                "time": ("time", np.array(
                    ["2020-01-01T00:00:00", "2020-01-01T00:30:00"],
                    dtype="datetime64[ns]")),
                "lat": ("time", [0.0, 1.0]),
                "lon": ("time", [0.0, 1.0]),
            })

        from typhon.files import FileInfo

        prim_infos = [FileInfo(f"/tmp/p{i}.nc") for i in range(3)]
        sec_infos = [FileInfo(f"/tmp/s{i}.nc") for i in range(3)]
        matches = [[p, [s]] for p, s in zip(prim_infos, sec_infos)]

        def fake_align(filesets, matches, return_info=True, compact=False,
                       skip_errors=False):
            for i in range(len(matches)):
                if i == 1:
                    # the corrupt file:
                    yield None
                else:
                    yield [prim_infos[i], make_ds()], [sec_infos[i], make_ds()]

        class FakePrimary:
            name = "primary"
            def align(self, other, matches=None, return_info=True,
                      compact=False, skip_errors=False):
                return fake_align(None, matches)

        class FakeSecondary:
            name = "secondary"

        def fake_collocate(*args, **kwargs):
            ds = xr.Dataset({
                "primary/lat": ("primary/time", [0.0]),
                "primary/lon": ("primary/time", [0.0]),
                "primary/time": ("primary/time", np.array(
                    ["2020-01-01T00:00:00"], dtype="datetime64[ns]")),
                "secondary/lat": ("secondary/time", [0.0]),
                "secondary/lon": ("secondary/time", [0.0]),
                "secondary/time": ("secondary/time", np.array(
                    ["2020-01-01T00:00:00"], dtype="datetime64[ns]")),
                "Collocations/pairs": (
                    ("pairs", "pair"), np.zeros((1, 2), dtype=int)),
                "Collocations/group": ("pairs", np.zeros(1, dtype=int)),
            })
            return ds

        collocator.collocate = fake_collocate
        collocator._debug = lambda msg: None

        from multiprocessing import Queue
        skipped_files_queue = Queue()

        results = list(collocator._collocate_matches(
            filesets=[FakePrimary(), FakeSecondary()],
            matches=matches,
            skip_file_errors=True,
            skipped_files_queue=skipped_files_queue,
            max_interval="30 min",
            max_distance="7.5 km",
        ))

        # One result per match; the corrupt one is skipped (None):
        assert len(results) == 3
        assert results[1][0] is None
        assert results[0][0] is not None
        assert results[2][0] is not None

        # The files of the skipped match are reported on the queue:
        skipped = []
        while True:
            try:
                skipped.append(skipped_files_queue.get(timeout=1))
            except Exception:
                break
        assert "/tmp/p1.nc" in skipped
        assert "/tmp/s1.nc" in skipped

    def test_collocate_filesets_reports_skipped_files(self, tmp_path):
        """collocate_filesets reports skipped files via skipped_files.

        When skip_file_errors is True and a file cannot be read, the file
        paths must be collected in collocator.skipped_files so that the
        caller can detect the data loss and, e.g., retry.
        """
        def make_ds():
            # 'time' is a variable on dimension 'scnline' (as for MHS/AVHRR),
            # so the dimension and coordinate names do not clash.
            return xr.Dataset({
                "time": ("scnline", np.array(
                    ["2020-01-01T00:00:00", "2020-01-01T00:30:00"],
                    dtype="datetime64[ns]")),
                "lat": ("scnline", [0.0, 1.0]),
                "lon": ("scnline", [0.0, 1.0]),
                "val": ("scnline", [1.0, 2.0]),
            })

        prim_dir = str(tmp_path / "primary")
        sec_dir = str(tmp_path / "secondary")
        import os
        os.makedirs(prim_dir)
        os.makedirs(sec_dir)

        prim = FileSet(
            name="primary",
            path=prim_dir + "/p.{year}{month}{day}{hour}{minute}{second}.nc",
            handler=NetCDF4(),
        )
        sec = FileSet(
            name="secondary",
            path=sec_dir + "/s.{year}{month}{day}{hour}{minute}{second}.nc",
            handler=NetCDF4(),
        )

        for hour in range(3):
            prim.write(
                make_ds(),
                os.path.join(prim_dir, f"p.20200101{hour:02d}0000.nc"),
            )
            sec.write(
                make_ds(),
                os.path.join(sec_dir, f"s.20200101{hour:02d}0000.nc"),
            )

        # Corrupt the secondary file for the second hour:
        with open(os.path.join(sec_dir, "s.20200101010000.nc"), "w") as f:
            f.write("this is not a netcdf file")

        collocator = Collocator()
        with warnings.catch_warnings():
            # The read failure warning is emitted in a worker process, so it
            # cannot be captured with pytest.warns:
            warnings.simplefilter("ignore", RuntimeWarning)
            list(collocator.collocate_filesets(
                [prim, sec],
                start="2020-01-01 00:00:00",
                end="2020-01-01 03:00:00",
                processes=2,
                max_interval="30 min",
                max_distance="7.5 km",
                skip_file_errors=True,
            ))

        # Both files of the skipped match are reported:
        skipped = collocator.skipped_files
        assert len(skipped) >= 2
        assert any("p.20200101010000.nc" in p for p in skipped), skipped
        assert any("s.20200101010000.nc" in p for p in skipped), skipped


