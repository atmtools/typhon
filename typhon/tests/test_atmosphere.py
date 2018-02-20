# -*- coding: utf-8 -*-
"""Testing the functions in typhon.atmosphere.
"""
import numpy as np

from typhon import atmosphere


class TestAtmosphere:
    """Testing the atmosphere functions."""
    def test_iwv(self):
        """Test the IWV calculation."""
        p = np.linspace(1000, 10, 10)
        T = 300 * np.ones(p.shape)
        z = np.linspace(0, 75000, 10)
        vmr = 0.1 * np.ones(p.shape)

        iwv = atmosphere.iwv(vmr, p, T, z)

        assert np.allclose(iwv, 27.3551036)

    def test_iwv_multi(self):
        """Test multidimensional IWV calculation."""
        p = np.linspace(1000, 10, 10)
        T = 300 * np.ones(p.shape)
        z = np.linspace(0, 75000, 10)
        vmr = 0.1 * np.ones((5, *p.shape))

        iwv = atmosphere.iwv(vmr, p, T, z, axis=1)

        assert np.allclose(iwv, np.repeat(27.3551036, 5))

    def test_relative_humidity(self):
        """Test the relative humidity calculation."""
        rh = atmosphere.relative_humidity(0.025, 1013e2, 300)

        assert np.allclose(rh, 0.7160499)

    def test_vmr(self):
        """Test the VMR calculation."""
        vmr = atmosphere.vmr(0.75, 1013e2, 300)

        assert np.allclose(vmr, 0.0261853)

    def test_vmr_rh_consistency(self):
        """Check the consistency of VMR and RH calculaion.

        Converting VMR into relative humidity and back to asure that both
        functions yield consistent results.
        """
        rh = atmosphere.relative_humidity(0.025, 1013e2, 300)
        vmr = atmosphere.vmr(rh, 1013e2, 300)

        assert np.allclose(vmr, 0.025)

    def test_moist_lapse_rate(self):
        """Test calculation of moist-adiabatic lapse rate."""
        gamma = atmosphere.moist_lapse_rate(1000e2, 300)

        assert np.isclose(gamma, 0.00367349)
