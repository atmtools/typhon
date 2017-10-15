# -*- coding: utf-8 -*-
"""Testing the functions in typhon.utils.
"""
import warnings

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
