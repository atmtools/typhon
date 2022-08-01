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

    def test_perfrequency2perwavelength(self):
        """Test conversion from a spectral quantities from per hz to per m."""
        f_grid = np.array([4e14, 6e14, 8e14])
        irrad = np.array([1e-12, 2e-12, 3e-12])
        x, _ = physics.perfrequency2perwavelength(irrad, f_grid)
        ref = np.array([6.40443063e9, 2.40166149e9, 5.33702552e8])
        assert np.allclose(x, ref)

    def test_perwavelength2perfrequency(self):
        """Test conversion from a spectral quantities from per m to per hz."""
        lam_grid = np.array([4e-7, 5e-7, 6e-7])
        irrad = np.array([1e9, 2e9, 3e9])
        x, _ = physics.perwavelength2perfrequency(irrad, lam_grid)
        ref = np.array([[3.60249223e-12, 1.66782048e-12, 5.33702552e-13]])
        assert np.allclose(x, ref)

    def test_perfreq_perlam_conversion(self):
        """Test conversion from a spectral quantities from per hz to per m and vice versa."""
        f_grid = np.array([4e14, 6e14, 8e14])
        irrad = np.array([1e-12, 2e-12, 3e-12])
        i_perm, lam_grid = physics.perfrequency2perwavelength(irrad, f_grid)
        i_perf, f_out = physics.perwavelength2perfrequency(i_perm, lam_grid)
        assert np.allclose(i_perf, irrad)
        assert np.allclose(f_out, f_grid)

    def test_perfrequency2perwavenumber(self):
        """Test conversion from a spectral quantities from per hz to per 1/m."""
        f_grid = np.array([4e14, 6e14, 8e14])
        irrad = np.array([1e-12, 2e-12, 3e-12])
        x, _ = physics.perfrequency2perwavenumber(irrad, f_grid)
        ref = np.array([0.00029979, 0.00059958, 0.00089938])
        assert np.allclose(x, ref)

    def test_perwavenumber2perfrequency(self):
        """Test conversion from a spectral quantities from per 1/m to per hz."""
        wn_grid = np.array([500, 1000, 1500])
        irrad = np.array([0.003, 0.004, 0.005])
        x, _ = physics.perwavenumber2perfrequency(irrad, wn_grid)
        ref = np.array([1.00069229e-11, 1.33425638e-11, 1.66782048e-11])
        assert np.allclose(x, ref) 

    def test_perfreq_perwn_conversion(self):
        """Test conversion from a spectral quantities from per hz to per 1/m and vice versa."""
        f_grid = np.array([4e14, 6e14, 8e14])
        irrad = np.array([1e-12, 2e-12, 3e-12])
        i_perwn, wn_grid = physics.perfrequency2perwavenumber(irrad, f_grid)
        i_perf, f_out = physics.perwavenumber2perfrequency(i_perwn, wn_grid)
        assert np.allclose(i_perf, irrad)
        assert np.allclose(f_out, f_grid)

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
