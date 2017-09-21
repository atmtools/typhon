# -*- coding: utf-8 -*-

"""Testing the functions in typhon.utils.
"""

import warnings

from nose.tools import raises

from typhon import utils


class TestUtils():
    """Testing the typhon.utils functions."""
    @raises(DeprecationWarning)
    def test_deprecated(self):
        """Test deprecation warning."""
        @utils.deprecated
        def func():
            pass

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            func()

    @raises(Exception)
    def test_image2mpeg(self):
        """Test the behavior when no files are found."""
        utils.image2mpeg(glob='', outfile='foo.mp4')
