# -*- coding: utf-8 -*-

"""Testing the functions in typhon.plots.
"""

import os

from typhon import plots


class TestPlots(object):
    """Testing the plot functions."""
    def test_figsize(self):
        """Test golden ratio for figures sizes."""
        ret = plots.figsize(10)

        assert ret == (10, 6.1803398874989481)

    def test_get_subplot_arrangement(self):
        """Test the determination of subplot arrangements."""
        shape = plots.get_subplot_arrangement(8)
        assert shape == (3, 3)

    def test_get_available_styles(self):
        """Check matplotlib stylesheet paths.

        This test checks the consinstency of the in and outputs
        of styles() and get_available_styles().
        """
        style_paths = [
            plots.styles(s) for s in plots.get_available_styles()]

        assert all(os.path.isfile(s) for s in style_paths)
