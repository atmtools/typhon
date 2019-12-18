import os
import tempfile

import numpy as np
import xarray as xr

from typhon.files import NetCDF4


class TestNetCDF4:
    def test_dimension_mapping(self):
        """
        If a subgroup has not defined a dimension, but its parent group has one
        with the same size and name, the subgroup should use that one.
        Otherwise it should use the one of the subgroup (test with dim1 in the
        root and subgroups).
        """
        fh = NetCDF4()

        with tempfile.TemporaryDirectory() as tdir:
            tfile = os.path.join(tdir, 'testfile')
            before = xr.Dataset({
                "var1": ("dim1", np.arange(5)),
                "group1/var1": ("group1/dim1", np.arange(5)),
                "group1/var2": ("group1/dim2", np.arange(5)),
                "group1/subgroup1/var1":
                    ("group1/subgroup1/dim1", np.arange(5)),
                "group1/subgroup1/var2":
                    ("group1/subgroup1/dim2", np.arange(5)),
                "group2/var1": ("group2/dim1", np.arange(5)),
                "group2/subgroup1/var1":
                    ("group1/subgroup1/dim1", np.arange(5)),
                "group3/var1": ("group3/dim1", np.arange(10)),
            }, coords={
                "dim1": ("dim1", np.arange(5)),
                "group1/dim1": ("group1/dim1", np.arange(5))
            })

            # Save the dataset and load it again:
            fh.write(before, tfile)
            after = fh.read(tfile)

            # How it should be after loading:
            check = xr.Dataset({
                "var1": ("dim1", np.arange(5)),
                "group1/var1": ("group1/dim1", np.arange(5)),
                "group1/var2": ("group1/dim2", np.arange(5)),
                "group1/subgroup1/var1": ("group1/dim1", np.arange(5)),
                "group1/subgroup1/var2": ("group1/dim2", np.arange(5)),
                "group2/var1": ("dim1", np.arange(5)),
                "group2/subgroup1/var1": ("dim1", np.arange(5)),
                "group3/var1": ("group3/dim1", np.arange(10)),
            }, coords={
                "dim1": ("dim1", np.arange(5)),
                "group1/dim1": ("group1/dim1", np.arange(5))
            })

        assert after.equals(check)

    def test_scalar_masked(self):
        """Test if scalar masked values read OK

        Test for issue #277
        """

        fh = NetCDF4()

        with tempfile.TemporaryDirectory() as tdir:
            tfile = os.path.join(tdir, "testfile.nc")
            before = xr.Dataset({"a": xr.DataArray(42)})
            before["a"].encoding = {"_FillValue": 42}
            fh.write(before, tfile)
            after = fh.read(tfile)
            assert np.isnan(after["a"]) # fill value should become nan

    def test_times(self):
        """Test if times are read correctly
        """

        fh = NetCDF4()

        with tempfile.TemporaryDirectory() as tdir:
            tfile = os.path.join(tdir, "testfile.nc")
            before = xr.Dataset(
                    {"a":
                        xr.DataArray(
                            np.array(
                                ["2019-02-14T09:00:00", "2019-02-14T09:00:01"],
                                dtype="M8[ns]"))})
            before["a"].encoding = {
                    "units": "seconds since 2019-02-14 09:00:00",
                    "scale_factor": 0.1}
            fh.write(before, tfile)
            after = fh.read(tfile)
            assert np.array_equal(before["a"], after["a"])

    def test_scalefactor(self):
        """Test if scale factors written/read correctly
        """

        fh = NetCDF4()

        with tempfile.TemporaryDirectory() as tdir:
            tfile = os.path.join(tdir, "testfile.nc")
            before = xr.Dataset(
                    {"a":
                        xr.DataArray(
                            np.array([0.1, 0.2]))})
            before["a"].encoding = {
                    "scale_factor": 0.1,
                    "_FillValue": 42,
                    "dtype": "int16"}
            fh.write(before, tfile)
            after = fh.read(tfile)
            assert np.allclose(before["a"], after["a"])
