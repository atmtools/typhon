# -*- coding: utf-8 -*-

"""Testing the functions in typhon.arts.
"""

import unittest
import shutil

from typhon import arts


class TestPlots(object):
    """Testing the plot functions."""
    @unittest.skipIf(not shutil.which('arts'), 'arts not in PATH')
    def test_run_arts(self):
        """Test ARTS system call.

        Note: This test is only run, if ARTS is found in PATH.
        """
        arts_out = arts.run_arts(help=True)

        assert arts_out.retcode == 0
