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
