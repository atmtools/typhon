# -*- coding: utf-8 -*-
"""
Implementation of scattering related types such as SingleScatteringData and
ScatteringMetaData.

"""

from __future__ import print_function

import copy
import numbers
from io import StringIO

import numpy as np

__all__ = ['SingleScatteringData',
           'ScatteringMetaData',
           ]

PARTICLE_TYPE_GENERAL = 10
PARTICLE_TYPE_MACROS_ISO = 20
PARTICLE_TYPE_HORIZ_AL = 30
PARTICLE_TYPE_SPHERICAL = 40

_old_ptype_mapping = {
    10: "general",
    20: "macroscopically_isotropic",
    30: "horizontally_aligned",
}

_valid_ptypes = {
    "general",
    "macroscopically_isotropic",
    "horizontally_aligned",
}


def dict_combine_with_default(in_dict, default_dict):
    """A useful function for dealing with dictionary function input.  Combines
    parameters from in_dict with those from default_dict with the output
    having default_dict values for keys not present in in_dict

    Args:
        in_dict (dict): Input dictionary.
        default_dict (dict): Dictionary with default values.

    Returns:
        dict: Dictionary with missing fields filled with default values.

    """
    if in_dict is None:
        out_dict = copy.deepcopy(default_dict)
    else:
        out_dict = copy.deepcopy(in_dict)
        for key in default_dict.keys():
            out_dict[key] = copy.deepcopy(in_dict.get(key, default_dict[key]))
    return out_dict


class SingleScatteringData:
    """The class representing the arts SingleScatteringData class.

    The data members of this object are identical to the class of the same name in
    ARTS; it includes all the single scattering properties required for
    polarized radiative transfer calculations: the extinction matrix, the
    phase matrix, and the absorption coefficient vector.  The angular,
    frequency, and temperature grids for which these are defined are also
    included.  Another data member - *ptype*, describes the orientational
    symmetry of the particle ensemble, which determines the format of the
    single scattering properties.  The data structure of the ARTS
    SingleScatteringData class is described in the ARTS User
    Guide.

    The methods in the SingleScatteringData class enable the
    calculation of the single scattering properties, and the output of the
    SingleScatteringData structure in the ARTS XML format (see example file).
    The low-level calculations are performed in arts_scat.
    """

    defaults = {'ptype': 'macroscopically_isotropic',
                # as defined in optproperties.h
                'description': 'SingleScatteringData created with Typhon.',
                'T_grid': np.array([250]),
                'za_grid': np.arange(0, 181, 10),
                'aa_grid': np.arange(0, 181, 10),
                }

    def __init__(self):
        pass

    @classmethod
    def from_data(cls, params=None, **kwargs):
        """ Constructor

        Parameters
        ----------

        ptype : string
            As for ARTS; see Arts User Guide

        f_grid : 1-D np.array
            np.array for frequency grid [Hz]

        T_grid : 1-D np.array
            np.array for temperature grid [K]

        za_grid : 1-D np.array
            np.array for zenith-angle grid [degree]

        aa_grid : 1-D np.array
            np.array for azimuth-angle grid [degree]

        Some inputs have default values, see SingleScatteringData.defaults.

        """
        obj = cls()

        # enable keyword arguments
        if kwargs and not params:
            params = kwargs

        params = dict_combine_with_default(params, obj.__class__.defaults)

        # check parameters
        # make sure grids are np np.arrays
        for grid in ['f', 'T', 'za', 'aa']:
            params[grid + '_grid'] = np.array(params[grid + '_grid'])

        if params['aspect_ratio'] == 1:
            raise ValueError(
                "'aspect_ratio' can not be set to exactly 1 due to numerical "
                "difficulties in the T-matrix code. use 1.000001 or 0.999999 "
                "instead.")

        if "description" in params:
            obj.description = params['description']
        else:
            obj.description = (
                obj.__class__.defaults['description'] +
                "module, which uses the T-matrix code of Mishchenko to calculate single \n"
                "scattering properties. The parameters used to create this file are shown\n"
                "below\n" + str(params))
            for k, v in params.items():
                setattr(obj, k, v)

    @property
    def ptype(self):
        """str: Particle type"""

        return self._ptype

    @ptype.setter
    def ptype(self, ptype):
        if type(ptype) is int:
            if ptype not in _old_ptype_mapping.keys():
                raise RuntimeError('Invalid ptype {}'.format(ptype))
            ptype = _old_ptype_mapping[ptype]
        else:
            if ptype not in _valid_ptypes:
                raise RuntimeError('Invalid ptype {}'.format(ptype))

        self._ptype = ptype

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a SingleScatteringData object from an xml.ElementTree.Element.
        """

        obj = cls()
        if 'version' in xmlelement.attrib.keys():
            version = int(xmlelement.attrib['version'])
        else:
            version = 1

        if version == 1:
            obj.ptype = int(xmlelement[0].value())
        else:
            obj.ptype = xmlelement[0].value()

        obj.description = xmlelement[1].value()
        obj.f_grid = xmlelement[2].value()
        obj.T_grid = xmlelement[3].value()
        obj.za_grid = xmlelement[4].value()
        obj.aa_grid = xmlelement[5].value()
        obj.pha_mat_data = xmlelement[6].value()
        obj.ext_mat_data = xmlelement[7].value()
        obj.abs_vec_data = xmlelement[8].value()
        obj.checksize()

        return obj

    def to_atmlab_dict(self):
        """Returns a copy of the SSD as a dictionary.

        Returns a dictionary compatible with an atmlab structure.

        Returns:
            Dictionary containing the grids and data.
        """

        d = {}
        d['ptype'] = self.ptype
        d['description'] = self.description
        d['f_grid'] = self.f_grid
        d['T_grid'] = self.T_grid
        d['za_grid'] = self.za_grid
        d['aa_grid'] = self.aa_grid
        d['pha_mat_data'] = self.pha_mat_data
        d['ext_mat_data'] = self.ext_mat_data
        d['abs_vec_data'] = self.abs_vec_data

        return d

    def write_xml(self, xmlwriter, attr=None):
        """Write a SingleScatterinData object to an ARTS XML file.
        """
        self.checksize()
        xmlwriter.open_tag("SingleScatteringData", attr)
        xmlwriter.write_xml(self.ptype)
        xmlwriter.write_xml(self.description)
        xmlwriter.write_xml(self.f_grid)
        xmlwriter.write_xml(self.T_grid)
        xmlwriter.write_xml(self.za_grid)
        xmlwriter.write_xml(self.aa_grid)
        xmlwriter.write_xml(self.pha_mat_data)
        xmlwriter.write_xml(self.ext_mat_data)
        xmlwriter.write_xml(self.abs_vec_data)
        xmlwriter.close_tag()

    def __repr__(self):
        S = StringIO()
        S.write("<SingleScatteringData ")
        S.write("ptype={} ".format(self.ptype))
        for nm in ("f_grid", "T_grid", "za_grid", "aa_grid"):
            g = getattr(self, nm)
            S.write(" ")
            if g.size > 1:
                S.write("%s=%4e..%4e" % (nm, g.min(), g.max()))
            elif g.size == 1:
                S.write("%s=%4e" % (nm, float(g.squeeze())))
            else:
                S.write("%s=[]" % nm)
        S.write(">")

        return S.getvalue()

    def normalise(self):
        """Normalises Z to E-A.
        """

        Z_int = self.integrate()
        E_min_A = self.ext_mat_data.squeeze() - self.abs_vec_data.squeeze()
        # use where to prevent divide by zero
        factor = E_min_A / np.where(Z_int == 0, 1, Z_int)
        factor.shape = (factor.size, 1, 1, 1, 1, 1)
        self.pha_mat_data[..., 0] *= factor
        # self.pha_mat_data[..., 0] = self.pha_mat_data[..., 0] * factor

    def normalised(self):
        """Returns normalised copy
        """

        c = copy.deepcopy(self)
        c.normalise()
        return c

    def __getitem__(self, v):
        """Get subset of single-scattering-data

        Must np.take four elements (f, T, za, aa).
        Only implemented for randomly oriented particles.
        """

        if self.ptype != PARTICLE_TYPE_MACROS_ISO:
            raise RuntimeError("Slicing implemented only for"
                               "ptype = %d. Found ptype = %d" %
                               (PARTICLE_TYPE_MACROS_ISO, self.ptype))
        v2 = list(v)
        for i, el in enumerate(v):
            # to preserve the rank of the data, [n] -> [n:n+1]
            if isinstance(el, numbers.Integral):
                v2[i] = slice(v[i], v[i] + 1, 1)
        f, T, za, aa = v2
        # make a shallow copy (view of the same data)
        c = copy.copy(self)
        c.f_grid = c.f_grid[f]
        c.T_grid = c.T_grid[T]
        c.za_grid = c.za_grid[za]
        c.aa_grid = c.aa_grid[aa]
        c.ext_mat_data = c.ext_mat_data[f, T, :, :, :]
        c.pha_mat_data = c.pha_mat_data[f, T, za, aa, :, :, :]
        c.abs_vec_data = c.abs_vec_data[f, T, :, :, :]
        c.checksize()
        return c

    def checksize(self):
        """Verifies size is consistent.

        Raises:
            RuntimeError

        """
        if not ((self.f_grid.size or 1, self.T_grid.size or 1) ==
                    self.ext_mat_data.shape[:2] ==
                    self.pha_mat_data.shape[:2] ==
                    self.abs_vec_data.shape[:2] and
                        (self.za_grid.size or 1, self.aa_grid.size or 1) ==
                        self.pha_mat_data.shape[2:4]):
            raise RuntimeError(
                "Inconsistent sizes in SingleScatteringData.\n"
                "f_grid: %s, T_grid: %s, za_grid: %s, aa_grid: %s, "
                "ext_mat: %s, pha_mat: %s, abs_vec: %s" %
                (self.f_grid.size or 1, self.T_grid.size or 1,
                 self.za_grid.size or 1, self.aa_grid.size or 1,
                 self.ext_mat_data.shape, self.pha_mat_data.shape,
                 self.abs_vec_data.shape))

    def assp2backcoef(self):
        """The function returns the radar backscattering coeffcient. This is the
        phase function times 4pi, following the standard definition in the radar
        community.

        Returns:
            Backscattering coefficients, one value for each frequency and
            temperature in S. [m2]

        """
        back_coef = np.multiply(4 * np.pi,
                                self.pha_mat_data[:, :, -1, 0, 0, 0, 0])
        return back_coef

    def assp2g(self):
        """For a normalised phase function (p), g equals the 4pi integral of
        p*cos(th), where th is the scattering angle. For pure isotropic
        scattering g = 0, while pure forward scattering has g=1.

        Warning, this function does not handle the extreme cases of
        delta-function type of forward or backward scattering lobes. A g of
        zero is returned for these cases.

        Returns:
            Backscattering coefficients, one value for each frequency and
            temperature in S. [m2]

        """
        g = np.zeros((len(self.f_grid), len(self.T_grid)))
        # ARTS uses pure phase matrix values, and not a normalised phase
        # function, and we need to include a normalisation.

        za_rad_grid = np.radians([self.za_grid])

        aziWeight = abs(np.sin(za_rad_grid))
        cosTerm = np.cos(za_rad_grid)

        for j in range(0, len(self.f_grid)):
            for i in range(0, len(self.T_grid)):
                phase_grid = self.pha_mat_data[j, i, :, 0, 0, 0, 0]

                normFac = np.trapz(np.multiply(
                    phase_grid, aziWeight), za_rad_grid)

                if normFac == 0:
                    # If normFac is zero, this means that phase_grid==0 and
                    # should indicate very small particles that have g=0.
                    g[j, i] = 0
                else:
                    temp_cosPhase = np.multiply(cosTerm, phase_grid)

                    temp = np.trapz(np.multiply(temp_cosPhase, aziWeight),
                                    za_rad_grid)

                    g[j, i] = np.divide(temp, normFac)
        return g

    def checkassp(self):
        """Verfies properties of SSP.

        Raises:
            PyARTSError: If ptype is not macroscopically isotropic, or if first
                and last value of za_grid does not equal exactly 0 and 180
                respectively.
        """

        if self.ptype != "macroscopically_isotropic":
            raise RuntimeError(
                "So far just complete random orientation is handled.")

        if self.za_grid[0] != 0:
            raise RuntimeError("First value of za_grid must be 0.")

        if self.za_grid[-1] != 180:
            raise RuntimeError("Last value of za_grid must be 180.")


class ScatteringMetaData:
    """Represents a ScatteringMetaData object.

    See online ARTS documentation for object details.

    """

    def __init__(self, description=None, source=None, refr_index=None,
                 mass=None, diameter_max=None, diameter_volume_equ=None,
                 diameter_area_equ_aerodynamical=None):

        self.description = description
        self.source = source
        self.refr_index = refr_index
        self.mass = mass
        self.diameter_max = diameter_max
        self.diameter_volume_equ = diameter_volume_equ
        self.diameter_area_equ_aerodynamical = diameter_area_equ_aerodynamical

    @property
    def description(self):
        """Free-form description of the scattering element, holding information
        deemed of interest by the user but not covered by other structure
        members (and not used within ARTS)."""
        return self._description

    @property
    def source(self):
        """Free-form description of the source of the data, e.g., Mie, T-Matrix,
        or DDA calculation or a database or a literature source."""
        return self._source

    @property
    def refr_index(self):
        """Free-form description of the underlying complex refractive index
        data, e.g., a literature source."""
        return self._refr_index

    @property
    def mass(self):
        """The mass of the scattering element."""
        return self._mass

    @property
    def diameter_max(self):
        """The maximum diameter (or dimension) of the scattering element,
        defined by the circumferential sphere diameter of the element. Note that
        this parameter is only used by some size distributions; it does not have
        a proper meaning if the scattering element represents an ensemble of
        differently sized particles."""
        return self._diameter_max

    @property
    def diameter_volume_equ(self):
        """The volume equivalent sphere diameter of the scattering element,
        i.e., the diameter of a sphere with the same volume. For nonspherical
        particles, volume refers to the volume of the particle-forming
        substance, not that of the circumferential sphere (which can be derived
        from diameter_max). If the particle consists of a mixture of materials,
        the substance encompasses the complete mixture. E.g., the substance of
        'soft' ice particles includes both the ice and the air."""
        return self._diameter_volume_equ

    @property
    def diameter_area_equ_aerodynamical(self):
        """The area equivalent sphere diameter of the scattering element, i.e.,
        the diameter of a sphere with the same cross-sectional area. Here, area
        refers to the aerodynamically relevant area, i.e., the cross-sectional
        area perpendicular to the direction of fall. Similarly to volume in the
        definition of diameter_volume_equ, for non-spherical and mixed-material
        particles, area refers to the area covered by the substance mixture of
        the particle."""
        return self._diameter_area_equ_aerodynamical

    @description.setter
    def description(self, description):
        if description is None:
            self._description = None
            return

        if type(description) is str:
            self._description = description
        else:
            raise TypeError('description has to be str.')

    @source.setter
    def source(self, source):
        if source is None:
            self._source = None
            return

        if type(source) is str:
            self._source = source
        else:
            raise TypeError('source has to be str.')

    @refr_index.setter
    def refr_index(self, refr_index):
        if refr_index is None:
            self._refr_index = None
            return

        if type(refr_index) is str:
            self._refr_index = refr_index
        else:
            raise TypeError('refr_index has to be str.')

    @mass.setter
    def mass(self, mass):
        if mass is None:
            self._mass = None
            return

        self._mass = mass

    @diameter_max.setter
    def diameter_max(self, diameter_max):
        if diameter_max is None:
            self._diameter_max = None
            return

        self._diameter_max = diameter_max

    @diameter_volume_equ.setter
    def diameter_volume_equ(self, diameter_volume_equ):
        if diameter_volume_equ is None:
            self._diameter_volume_equ = None
            return

        self._diameter_volume_equ = diameter_volume_equ

    @diameter_area_equ_aerodynamical.setter
    def diameter_area_equ_aerodynamical(self, diameter_area_equ_aerodynamical):
        if diameter_area_equ_aerodynamical is None:
            self._diameter_area_equ_aerodynamical = None
            return

        self._diameter_area_equ_aerodynamical = diameter_area_equ_aerodynamical

    def __repr__(self):
        s = StringIO()
        s.write("<ScatteringMetaData ")
        s.write("description=%s " % self.description)
        s.write("source=%s " % self.source)
        s.write("refr_index=%s " % self.refr_index)
        s.write("mass=%4e " % self.mass)
        s.write("diameter_max=%4e " % self.diameter_max)
        s.write("diameter_volume_equ=%4e " % self.diameter_volume_equ)
        s.write(
            "diameter_area_equ_aerodynamical=%4e " % self.diameter_area_equ_aerodynamical)
        return s.getvalue()

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a ScatteringMetaData object from an existing file.
        """
        if 'version' in xmlelement.attrib.keys():
            version = int(xmlelement.attrib['version'])
        else:
            version = 1

        if version != 3:
            raise Exception('Unsupported ScatteringMetaData version '
                            '{}. Version must be 3.'.format(version))

        obj = cls()
        obj.description = xmlelement[0].value()
        obj.source = xmlelement[1].value()
        obj.refr_index = xmlelement[2].value()
        obj.mass = xmlelement[3].value()
        obj.diameter_max = xmlelement[4].value()
        obj.diameter_volume_equ = xmlelement[5].value()
        obj.diameter_area_equ_aerodynamical = xmlelement[6].value()
        return obj

    def to_atmlab_dict(self):
        """Returns a copy of the SMD as a dictionary.

        Returns a dictionary compatible with an atmlab structure.

       Returns:
           Dictionary containing the grids and data.
        """

        d = {}
        d['version'] = 3
        d['description'] = self.description
        d['source'] = self.source
        d['refr_index'] = self.refr_index
        d['mass'] = self.mass
        d['diameter_max'] = self.diameter_max
        d['diameter_volume_equ'] = self.diameter_volume_equ
        d['diameter_area_equ_aerodynamical'] = self.diameter_area_equ_aerodynamical

        return d

    def write_xml(self, xmlwriter, attr=None):
        """Write a ScatteringMetaData object to an ARTS XML file.
        """
        if attr is None:
            attr = {}
        attr['version'] = 3

        xmlwriter.open_tag("ScatteringMetaData", attr)
        xmlwriter.write_xml(self.description)
        xmlwriter.write_xml(self.source)
        xmlwriter.write_xml(self.refr_index)
        xmlwriter.write_xml(self.mass)
        xmlwriter.write_xml(self.diameter_max)
        xmlwriter.write_xml(self.diameter_volume_equ)
        xmlwriter.write_xml(self.diameter_area_equ_aerodynamical)
        xmlwriter.close_tag()
