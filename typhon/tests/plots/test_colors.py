# -*- coding: utf-8 -*-
"""Testing the functions in typhon.plots.colors.
"""
import filecmp
import os
from tempfile import mkstemp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pytest

from typhon.plots import colors


class TestColors:
    """Testing the cm functions."""
    ref_dir = os.path.join(os.path.dirname(__file__), "reference", "")

    def setup_method(self):
        """Create a temporary file."""
        fd, self.f = mkstemp()
        os.close(fd)

    def teardown_method(self):
        """Delete temporary file."""
        os.remove(self.f)

    def test_cmap2rgba(self):
        """Check colormap to RGB conversion."""
        ref = np.loadtxt(os.path.join(self.ref_dir, 'viridis.txt'),
                         comments='%')
        rgb = colors.cmap2rgba('viridis', 256)[:, :3]  # ignore alpha

        assert np.allclose(ref, rgb, atol=0.001)

    def test_cmap2rgba_interpolation(self):
        """Check colormap to RGBA interpolation."""
        max_planck_duplicates = np.array([
            [0., 0.4627451, 0.40784314, 1.],
            [0.48235294, 0.70980392, 0.67843137, 1.],
            [0.74901961, 0.85098039, 0.83137255, 1.],
            [0.96078431, 0.97254902, 0.97647059, 1.],
            [0.96078431, 0.97254902, 0.97647059, 1.],
        ])

        max_planck_interpolated = np.array([
            [0., 0.4627451, 0.40784314, 1.],
            [0.36318339, 0.64876586, 0.61158016, 1.],
            [0.6172549, 0.7812226, 0.75580161, 1.],
            [0.8038293, 0.88244521, 0.86892734, 1.],
            [0.96078431, 0.97254902, 0.97647059, 1.],
        ])

        assert np.allclose(
            max_planck_interpolated,
            colors.cmap2rgba('max_planck', 5, interpolate=True)
        )

        assert np.allclose(
            max_planck_duplicates,
            colors.cmap2rgba('max_planck', 5, interpolate=False)
        )

    def test_cmap2cpt(self):
        """Export colormap to cpt file."""
        colors.cmap2cpt('viridis', filename=self.f)
        ref = os.path.join(self.ref_dir, 'viridis.cpt')

        with open(self.f) as testfile, open(ref) as reffile:
            assert testfile.readlines() == reffile.readlines()

    def test_cmap2txt(self):
        """Export colormap to txt file."""
        colors.cmap2txt('viridis', filename=self.f)
        ref = os.path.join(self.ref_dir, 'viridis.txt')

        with open(self.f) as testfile, open(ref) as reffile:
            assert testfile.readlines() == reffile.readlines()

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
        assert np.allclose(viridis(idx), cmap(idx), atol=0.001)

    def test_cmap_from_act(self):
        """Import colormap from act file."""
        viridis = plt.get_cmap('viridis')
        cmap = colors.cmap_from_act(os.path.join(self.ref_dir, 'viridis.act'))

        plt.register_cmap(cmap=viridis)  # Register original viridis.

        idx = np.linspace(0, 1, 256)
        assert np.allclose(viridis(idx), cmap(idx), atol=0.004)

    def test_get_material_design(self):
        """Test the retrieval of material design colors."""
        hex_color = colors.get_material_design('red', shade='500')
        assert hex_color == '#F44336'

        hex_colors = colors.get_material_design('red', shade=None)
        assert hex_colors == ['#FFEBEE', '#FFCDD2', '#EF9A9A', '#E57373',
                              '#EF5350', '#F44336', '#E53935', '#D32F2F',
                              '#C62828', '#B71C1C', '#FF8A80', '#FF5252',
                              '#FF1744', '#D50000']

    def test_get_material_design_valuerror(self):
        """Test the behavior for undefined material design colors or shades."""
        with pytest.raises(ValueError):
            colors.get_material_design('undefined_color')

        with pytest.raises(ValueError):
            colors.get_material_design('red', 'undefined_shade')

    def test_named_color_mapping(self):
        """Test if the typhon colors are available in the name mapping."""
        assert all([c in mcolors.get_named_colors_mapping()
                   for c in colors.TYPHON_COLORS.keys()])

    def test_named_color_hex(self):
        """Test if the 'ty:uhh-red' hex-value is correct."""
        assert mcolors.get_named_colors_mapping()['ty:uhh-red'] == '#ee1d23'
