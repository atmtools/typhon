# -*- coding: utf-8 -*-

"""Testing the functions in typhon.plots.colors.
"""

import filecmp
import matplotlib.pyplot as plt
import numpy as np
import os
from tempfile import mkstemp

from typhon.plots import colors


class TestColors(object):
    """Testing the cm functions."""
    ref_dir = os.path.join(os.path.dirname(__file__), "reference", "")

    def setUp(self):
        """Create a temporary file."""
        _, self.f = mkstemp()

    def tearDown(self):
        """Delete temporary file."""
        os.remove(self.f)

    def test_cmap2cpt(self):
        """Export colormap to cpt file."""
        colors.cmap2cpt('viridis', filename=self.f)
        ref = os.path.join(self.ref_dir, 'viridis.cpt')

        assert filecmp.cmp(self.f, ref)

    def test_cmap2txt(self):
        """Export colormap to txt file."""
        colors.cmap2txt('viridis', filename=self.f)
        ref = os.path.join(self.ref_dir, 'viridis.txt')

        assert filecmp.cmp(self.f, ref)

    def test_cmap2act(self):
        """Export colormap to act file."""
        colors.cmap2act('viridis', filename=self.f)
        ref = os.path.join(self.ref_dir, 'viridis.act')

        assert filecmp.cmp(self.f, ref)

    def test_cmap_from_txt(self):
        """Import colormap from txt file."""
        viridis = plt.get_cmap('viridis')
        cmap = colors.cmap_from_txt(os.path.join(self.ref_dir, 'viridis.txt'))

        plt.register_cmap(cmap=viridis)  # Register original viridis.

        idx = np.linspace(0, 1, 256)
        assert np.allclose(viridis(idx), cmap(idx))

    def test_cmap_from_act(self):
        """Import colormap from act file."""
        viridis = plt.get_cmap('viridis')
        cmap = colors.cmap_from_act(os.path.join(self.ref_dir, 'viridis.act'))

        plt.register_cmap(cmap=viridis)  # Register original viridis.

        idx = np.linspace(0, 1, 256)
        assert np.allclose(viridis(idx), cmap(idx), atol=0.004)

    def test_mpl_colors(self):
        """Check colormap to RGB conversion."""
        ref = np.loadtxt(os.path.join(self.ref_dir, 'viridis.txt'),
                         comments='%')
        rgb = colors.mpl_colors('viridis', 256)[:, :3]  # ignore alpha

        assert np.allclose(ref, rgb)
