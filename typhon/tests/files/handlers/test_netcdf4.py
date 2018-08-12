import tempfile

import numpy as np
from typhon.files import NetCDF4
import xarray as xr


class TestNetCDF4:
    def test_dimension_mapping(self):
        """
        If a subgroup has not defined a dimension, but its parent group has one
        with the same size and name, the subgroup should use that one.
        Otherwise it should use the one of the subgroup (test with dim1 in the
        root and subgroups).
        """
        fh = NetCDF4()

        with tempfile.NamedTemporaryFile() as file:
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
            fh.write(before, file.name)
            after = fh.read(file.name)

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
