import numpy as np
from typhon.spareice import Array, GroupedArrays
import xarray as xr


class TestArray:
    """Testing the array methods."""
    pass


class TestGroupedArrays:
    """Testing the GroupedArrays methods."""

    def test_dict(self):
        # TODO: Implement test for export to / import from python dictionary
        # TODO: objects
        pass

    def test_equal(self):
        a = GroupedArrays()
        a["group1/group2/data"] = np.arange(1, 100).astype(float)
        a["group1/data"] = np.arange(1, 10).astype(float)
        a["data"] = np.arange(-30, 10).astype(float)

        # Check whether it does find equality:
        assert a == a

        # Check whether it does find inequality because the variables have
        # different lengths:
        assert a != a[:5]

        # Check whether it does find inequality because a variable's content is
        # different but has the same length:
        b = a.copy()
        b["data"] = np.arange(-20, 20).astype(float)
        assert a != b

        # Check whether it does find inequality (if one variable does not exist
        # in the other group):
        del b["data"]
        assert a != b

    def test_xarray(self):
        # TODO: Implement test for export to / import from xarray.Dataset
        pass


