# -*- coding: utf-8 -*-
"""Testing the functions in typhon.plots.
"""
import os
import pathlib

import pytest

from unittest import mock

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

    @mock.patch("lzma.open", autospec=True)
    @mock.patch("pickle.dump", autospec=True)
    def test_write_multi(self, pd, lo):
        fig = mock.MagicMock()
        plots.common.write_multi(fig, "/tmp/nothing")
        fig.canvas.print_figure.assert_any_call(
                pathlib.Path("/tmp/nothing.png"))
        fig.canvas.print_figure.assert_any_call(
                pathlib.Path("/tmp/nothing.pdf"))
        assert fig.canvas.print_figure.call_count == 2
        lo.assert_called_once_with(
                pathlib.Path("/tmp/nothing.pkz"), "wb")
        pd.assert_called_once()
