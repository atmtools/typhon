# -*- coding: utf-8 -*-
"""Testing the functions in typhon.plots.
"""
import os

import pytest

from typhon import plots


class TestPlots:
    """Testing the plot functions."""
    def test_figsize(self):
        """Test golden ratio for figures sizes."""
        ret = plots.figsize(10)

        assert ret == (10, 6.1803398874989481)

    def test_get_subplot_arrangement(self):
        """Test the determination of subplot arrangements."""
        shape = plots.get_subplot_arrangement(8)
        assert shape == (3, 3)

    def test_get_style_path_method(self):
        assert os.path.isfile(plots.styles.get('typhon'))

    def test_get_style_path_call(self):
        assert os.path.isfile(plots.styles('typhon'))

    def test_get_style_path_default(self):
        assert os.path.isfile(plots.styles())

    def test_undefined_style(self):
        with pytest.raises(ValueError):
            plots.styles.get('Undefined Stylesheet Name')

    def test_available_styles(self):
        style_paths = plots.styles.available

        assert isinstance(style_paths, list)
        assert len(style_paths) > 0
        assert all(os.path.isfile(plots.styles(s)) for s in style_paths)
