# -*- coding: utf-8 -*-
"""Testing the functions in typhon.physics.
"""
import numpy as np

from typhon import physics


class TestEM:
    """Testing the typhon.physics.em functions."""
    def test_planck(self):
        """Test calculation of Planck black body radiation."""
        x = physics.planck(183e9, 273.15)
        assert np.allclose(x, 2.7655049656659619e-15)

    def test_planck_wavelength(self):
        """Test calculation of Planck black body radiation (for wavelength)."""
        x = physics.planck_wavelength(0.01, 273.15)
        assert np.allclose(x, 0.00022552294907196693)

    def test_planck_wavenumber(self):
        """Test calculation of Planck black body radiation (for wavenumber)."""
        x = physics.planck_wavenumber(100, 273.15)
        assert np.allclose(x, 2.2552294907196695e-08)

    def test_rayleighjeans(self):
        """Test calculation Rayleigh-Jeans black body radiation."""
        x = physics.rayleighjeans(183e9, 273.15)
        assert np.allclose(x, 2.810445098218606e-15)

    def test_rayleighjeans_wavelength(self):
        """Test calculation RJ black body radiation (for wavelength)."""
        x = physics.rayleighjeans_wavelength(0.01, 273.15)
        assert np.allclose(x, 0.00022611794774492813)

    def test_radiance2planckTb(self):
        """Test conversion between spectral radiance and Planck Tb."""
        f = np.linspace(100, 80e12, 100)
        r = physics.planck(f, 300)

        tb = physics.radiance2planckTb(f, r)

        assert np.allclose(tb, 300)

    def test_radianc2rayleighjeansTb(self):
        """Test conversion between spectral radiance and Rayleigh-Jeans Tb."""
        f = np.linspace(100, 80e12, 100)
        r = physics.rayleighjeans(f, 300)

        tb = physics.radiance2rayleighjeansTb(f, r)

        assert np.allclose(tb, 300)

    def test_snell(self):
        """Test Snell's law."""
        x = physics.snell(1, 1, 60)
        assert np.allclose(x, 60)

    def test_fresnel(self):
        """Test Fresnel law."""
        x = physics.fresnel(0.6, 1, 45)
        ref = (0.13098189212919023, -0.3619142054813409)
        assert np.allclose(x, ref)

    def test_frequency2wavelength(self):
        """Test conversion from frequency to wavelength."""
        x = physics.frequency2wavelength(183e9)
        assert np.allclose(x, 0.0016382101530054644)

    def test_frequency2wavenumber(self):
        """Test conversion from frequency to wavenumber."""
        x = physics.frequency2wavenumber(183e9)
        assert np.allclose(x, 610.4222942126182)

    def test_wavelength2frequency(self):
        """Test conversion from wavelength to frequency."""
        x = physics.wavelength2frequency(0.01)
        assert np.allclose(x, 29979245800.0)

    def test_wavelength2wavenumber(self):
        """Test conversion from wavelength to wavenumber."""
        x = physics.wavelength2wavenumber(0.01)
        assert x == 100

    def test_wavenumber2frequency(self):
        """Test conversion from wavenumber to frequency."""
        x = physics.wavelength2frequency(100)
        assert np.allclose(x, 2997924.58)

    def test_wavenumber2wavelength(self):
        """Test conversion from wavenumber to wavelength."""
        x = physics.wavelength2wavenumber(100)
        assert x == 0.01


class TestThermodynamics:
    """Testing the typhon.physics.thermodynamics functions."""
    def test_e_eq_ice_mk(self):
        """Test calculation of equilibrium water vapor pressure over ice."""
        x = physics.e_eq_ice_mk(260)
        assert np.allclose(x, 195.81934571953161)

    def test_e_eq_water_mk(self):
        """Test calculation of equilibrium water vapor pressure over water."""
        x = physics.e_eq_water_mk(280)
        assert np.allclose(x, 991.85662101784112)

    def test_e_eq_mixed_mk(self):
        """Test calculation of vapor pressure with respect to mixed phase."""
        assert np.allclose(physics.e_eq_ice_mk(240),
                           physics.e_eq_mixed_mk(240))

        assert np.allclose(physics.e_eq_water_mk(290),
                           physics.e_eq_mixed_mk(290))

        assert (physics.e_eq_ice_mk(263)
                < physics.e_eq_mixed_mk(263)
                < physics.e_eq_water_mk(263))

    def test_density(self):
        """Test calculation of air density."""
        x = physics.density(1013.15e2, 273.15)
        assert np.allclose(x, 1.2921250376634072)

    def test_mixing_ratio2specific_humidity(self):
        """Test conversion of mass mixing ratio to specific humidity."""
        q = physics.mixing_ratio2specific_humidity(0.02)
        assert np.isclose(q, 0.0196078431372549)

    def test_mixing_ratio2vmr(self):
        """Test conversion of mass mixing ratio to VMR."""
        x = physics.mixing_ratio2vmr(0.02)
        assert np.isclose(x, 0.03115371853180794)

    def test_specific_humidity2mixing_ratio(self):
        """Test conversion of specific humidity to mass mixing ratio."""
        w = physics.specific_humidity2mixing_ratio(0.02)
        assert np.isclose(w, 0.020408163265306124)

    def test_specific_humidity2vmr(self):
        """Test conversion of specific humidity to VMR."""
        x = physics.specific_humidity2vmr(0.02)
        assert np.isclose(x, 0.03176931009073226)

    def test_vmr2mixing_ratio(self):
        """Test conversion of VMR to mass mixing ratio."""
        w = physics.vmr2mixing_ratio(0.04)
        assert np.isclose(w, 0.025915747437955664)

    def test_vmr2specific_humidity(self):
        """Test conversion of VMR to specific humidity."""
        q = physics.vmr2specific_humidity(0.04)
        assert np.isclose(q, 0.025261087474946833)
