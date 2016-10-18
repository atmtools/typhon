# -*- coding: utf-8 -*-

"""Testing the functions in typhon.cm.
"""

from tempfile import mkstemp
import numpy as np
import os

from typhon import cm


class TestCM(object):
    """Testing the cm functions."""
    ref_dir = os.path.join(os.path.dirname(__file__), "reference", "")

    def setUp(self):
        """Create a temporary file."""
        _, self.f = mkstemp()

    def tearDown(self):
        """Delete temporary file."""
        os.remove(self.f)

    def test_cmap2cpt(self):
        """Export colormap to cpt file.

        The created file is compare line by line (timestamps are ignored).
        """
        cm.cmap2cpt('viridis', filename=self.f)
        ref = os.path.join(self.ref_dir, 'viridis.cpt')

        same = True

        with open(self.f) as f1, open(ref) as f2:
            for l1, l2 in zip(f1.readlines(), f2.readlines()):
                if 'Created' not in l1 and l1 != l2:
                    same = False
                    break

        assert same

    def test_cmap2txt(self):
        """Export colormap to txt file.

        The created file is compare line by line (timestamps are ignored).
        """
        cm.cmap2txt('viridis', filename=self.f)
        ref = os.path.join(self.ref_dir, 'viridis.txt')

        same = True

        with open(self.f) as f1, open(ref) as f2:
            for l1, l2 in zip(f1.readlines(), f2.readlines()):
                if 'Created' not in l1 and l1 != l2:
                    same = False
                    break

        assert same

    def test_mpl_colors(self):
        """Check colormap to RGB conversion."""
        ref = np.loadtxt(os.path.join(self.ref_dir, 'viridis.txt'),
                         comments='%')
        rgb = cm.mpl_colors('viridis', 256)[:, :3]  # ignore alpha

        assert np.allclose(ref, rgb)
