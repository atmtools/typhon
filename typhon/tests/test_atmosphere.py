# -*- coding: utf-8 -*-

"""Testing the functions in typhon.atmosphere.
"""

import numpy as np

from typhon import atmosphere


class TestAtmosphere(object):
    """Testing the atmosphere functions."""
    def test_iwv(self):
        """Test the IWV calculation."""
        p = np.linspace(1000, 10, 10)
        T = 300 * np.ones(p.shape)
        z = np.linspace(0, 75000, 10)
        vmr = 0.1 * np.ones(p.shape)

        iwv = atmosphere.iwv(vmr, p, T, z)

        assert np.allclose(iwv, 27.355103695371774)
