# -*- coding: utf-8 -*-
"""Testing the functions in typhon.atmosphere.
"""
import numpy as np
import pytest

from typhon.physics import atmosphere


class TestAtmosphere:
    """Testing the atmosphere functions."""
    def test_integrate_water_vapor_hse(self):
        """Test the IWV calculation in hydrostatic equilibrium."""
        p = np.linspace(1000e2, 500e2, 10)
        vmr = np.linspace(0.025, 0.0025, p.size)

        iwv = atmosphere.integrate_water_vapor(vmr, p)

        assert np.allclose(iwv, 43.8845)

    def test_integrate_water_vapor_nonhse(self):
        """Test the IWV calculation in non-hydrostatic atmosphere."""
        p = np.linspace(1000e2, 500e2, 10)
        T = 288 * np.ones(p.size)
        z = np.linspace(0, 5000, p.size)
        vmr = np.linspace(0.025, 0.0025, p.size)

        iwv = atmosphere.integrate_water_vapor(vmr, p, T, z)

        assert np.allclose(iwv, 42.4062)

    def test_integrate_water_vapor_ivalidin(self):
        """Test invalid number of arguments to IWV calculation."""
        dummy = np.ones(10)

        with pytest.raises(ValueError):
            atmosphere.integrate_water_vapor(dummy, dummy, dummy)

    def test_integrate_water_vapor_multi(self):
        """Test multidimensional IWV calculation."""
        p = np.linspace(1000e2, 500e2, 10)
        vmr = np.linspace(0.025, 0.0025, p.size)
        vmr_multi = np.repeat(vmr[np.newaxis, :], 5, axis=0)

        iwv = atmosphere.integrate_water_vapor(vmr_multi, p, axis=1)

        assert np.allclose(iwv, np.repeat(43.8845, 5))

    def test_vmr2relative_humidity(self):
        """Test conversion from VMR into relative humidity."""
        rh = atmosphere.vmr2relative_humidity(0.025, 1013e2, 300)

        assert np.allclose(rh, 0.7160499)

    def test_relative_humidity2vmr(self):
        """Test conversion from relative humidity into VMR."""
        vmr = atmosphere.relative_humidity2vmr(0.75, 1013e2, 300)

        assert np.allclose(vmr, 0.0261853)

    def test_vmr_rh_conversion(self):
        """Check the consistency of VMR and relative humidity conversion.

        Converting VMR into relative humidity and back to ensure that both
        functions yield consistent results.
        """
        rh = atmosphere.vmr2relative_humidity(0.025, 1013e2, 300)
        vmr = atmosphere.relative_humidity2vmr(rh, 1013e2, 300)

        assert np.allclose(vmr, 0.025)

    def test_moist_lapse_rate(self):
        """Test calculation of moist-adiabatic lapse rate."""
        gamma = atmosphere.moist_lapse_rate(1000e2, 300)

        assert np.isclose(gamma, 0.00367349)

    def test_standard_atmosphere(self):
        """Test International Standard Atmosphere."""
        isa = atmosphere.standard_atmosphere

        assert np.isclose(isa(0), 288.1831)  # Surface temperature
        assert np.isclose(isa(81e3), 194.5951)  # Test extrapolation
        # Test call with ndarray.
        assert np.allclose(isa(np.array([0, 15e3])),
                           np.array([288.1831, 216.65]))

    def test_standard_atmosphere_pressure(self):
        """Test International Standard Atmosphere."""
        isa = atmosphere.standard_atmosphere

        assert np.isclose(isa(1000e2, coordinates='pressure'), 288.0527)

    def test_pressure2height(self):
        """Test conversion from atmospheric pressure to height."""
        p = np.array([1000e2, 750e2, 500e2, 100e2])

        z = atmosphere.pressure2height(p)
        z_ref = np.array([0., 2358.129, 5473.647, 15132.902])

        assert np.allclose(z, z_ref)

    def test_pressure2height_with_T(self):
        """Test conversion from atmospheric pressure to height."""
        p = np.array([1000e2, 750e2, 500e2, 100e2])
        T = np.array([288, 275, 255, 215])

        z = atmosphere.pressure2height(p, T=T)
        z_ref = np.array([0., 2360.809, 5482.749, 15135.793])

        assert np.allclose(z, z_ref)
