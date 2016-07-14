# -*- coding: utf-8 -*-

"""Testing the basic plotting functions.

This module provides basic functions to test plotting utilities.
"""

import os
import glob

import numpy as np
import matplotlib as mpl

import typhon.plots


class TestPlots(object):
    """Testing plotting functions."""

    def test_install_mplstyles(self):
        """Install matplotlib style sheets."""
        config_dir = os.path.join(mpl.get_configdir(), 'stylelib')

        if os.path.isdir(config_dir):
            created_dir = False
        else:
            os.mkdir(config_dir)
            created_dir = True

        typhon_styles = [
            os.path.basename(s)
            for s
            in glob.glob(os.path.join('..', 'stylelib', '*.mplstyle'))
            ]

        before = set(os.listdir(config_dir))

        typhon.plots.install_mplstyles()

        after = set(os.listdir(config_dir))

        for sl in after - before:
            if sl in typhon_styles:
                os.remove(os.path.join(config_dir, sl))

        if created_dir:
            os.rmdir(config_dir)

        assert set(typhon_styles).issubset(set(after))
