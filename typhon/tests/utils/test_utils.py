# -*- coding: utf-8 -*-
"""Testing the functions in typhon.utils.
"""
import warnings
import numpy
import xarray

import pytest

from typhon import utils


class TestUtils():
    """Testing the typhon.utils functions."""
    def test_deprecated(self):
        """Test deprecation warning."""
        @utils.deprecated
        def func():
            pass

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(DeprecationWarning):
                func()

    def test_image2mpeg(self):
        """Test the behavior when no files are found."""
        with pytest.raises(Exception):
            utils.image2mpeg(glob='', outfile='foo.mp4')

    def test_undo_xarray_floatification(self):
        ds = xarray.Dataset(
            {"a": (["x"], numpy.array([1, 2, 3], dtype="f4")),
             "b": (["x"], numpy.array([2.0, 3.0, 4.0]))})
        ds["a"].encoding = {"dtype": numpy.dtype("i4"),
                            "_FillValue": 1234}
        ds2 = utils.undo_xarray_floatification(ds)
        assert ds["a"].encoding == ds2["a"].encoding
        assert numpy.allclose(ds["a"], ds2["a"])
        assert ds2["a"].dtype == ds2["a"].encoding["dtype"]
