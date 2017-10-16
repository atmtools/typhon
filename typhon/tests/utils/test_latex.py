# -*- coding: utf-8 -*-
"""Testing the functions in typhon.latex.
"""
import filecmp
import os
from tempfile import mkstemp

import numpy as np

from typhon import latex


class TestLaTeX:
    """Testing the latex functions."""
    ref_dir = os.path.join(os.path.dirname(__file__), "reference", "")

    def setup_method(self):
        """Create a temporary file."""
        _, self.f = mkstemp()

    def teardown_method(self):
        """Delete temporary file."""
        os.remove(self.f)

    def test_texify_matrix(self):
        """Save NumPy array as LaTeX table."""
        latex.texify_matrix(
            np.arange(20).reshape(5, 4),
            fmt="%.3f",
            filename=self.f,
            caption="This is a test caption.",
            heading=['H1', 'H2', 'H3', 'H4'],
            align='c',
            delimiter=False
            )

        assert filecmp.cmp(self.f, os.path.join(self.ref_dir, 'matrix.tex'))
