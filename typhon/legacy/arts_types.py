#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The arts_types module includes support for various ARTS-classes.

Due to the dynamic nature of Python, some are implemented in a more
generic way. For example, ArrayOf can be easily subclassed to be an np.array
of anything, and all gridded-fields are subclasses of GriddedField.

Classes of special interest may be:

- LatLonGriddedField3: Special case of GriddedField3 for lat/lon/pressure
  data
- AbsSpecies
- SingleScatteringData
- ScatteringMetaData

This module allows the generation, manipulation, and input/output in ARTS
XML format of these objects.
"""

from __future__ import print_function

import copy
import gzip
import numbers

import numpy as np

try:
    from cStringIO import StringIO
except:
    from io import StringIO

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    unicode = str
    basestring = (str, bytes)

from . import artsXML
from . import general
from . import arts_math

PARTICLE_TYPE_GENERAL = 10
PARTICLE_TYPE_MACROS_ISO = 20
PARTICLE_TYPE_HORIZ_AL = 30
PARTICLE_TYPE_SPHERICAL = 40


class BadFieldDimError(general.PyARTSError, ValueError): pass


class ArtsType(object):
    def save(self, f, compressed=None):
        """Writes data to file
        """

        c = self.to_xml()

        if isinstance(f, basestring):
            if compressed is None:
                compressed = f.lower().endswith(".gz")
            if compressed:
                writer = gzip.GzipFile(f, "wb")
            else:
                writer = open(f, "wb")
        else:
            writer = f
        with artsXML.XMLfile(writer, header=True) as out:
            out.write(c)


class ArrayOf(ArtsType, list):
    """Represents an np.array of somethinge

    The type is np.taken from the 'contains' attribute, to be defined by
    subclasses.
    """

    contains = None

    def to_xml(self):
        xml_obj = artsXML.XML_Obj(tag="Array",
                                  attributes={
                                      "type": getattr(self.contains, "name",
                                                      self.contains.__name__),
                                      "nelem": len(self)})
        for elem in self:
            xml_obj.write(elem.to_xml())
        return xml_obj.finalise().str

    def append(self, obj):
        if not isinstance(obj, self.contains):
            raise TypeError("Wrong type in np.array. Expected: %s. Got: %s" %
                            (self.contains, type(obj)))
        super(ArrayOf, self).append(obj)

    def __repr__(self):
        return ("<" + self.__class__.__name__ + "\n["
                + ",\n ".join(repr(elem) for elem in self) + "]" + ">")

    def __getattr__(self, attr):
        L = []
        for elem in self:
            L.append(getattr(elem, attr))
        return L

    @classmethod
    def load(cls, f):
        """Loads ArrayOfSomething

        - f: file to read it from

        Returns ArrayOfSomething
        """
        filedata = artsXML.load(f)
        obj = cls()

        tp = cls.contains
        tpname = cls.contains.__name__
        for artsXML_object in filedata:
            elem = tp._load_from_artsXML_object(artsXML_object[tpname])
            obj.append(elem)
        return obj


class SingleScatteringData(ArtsType):
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

    Constructor input
    ~~~~~~~~~~~~~~~~~
    
    ptype : integer
        As for ARTS; see Arts User Guide

    f_grid : 1-D np.array
        np.array for frequency grid [Hz]

    T_grid : 1-D np.array
        np.array for temperature grid [K]

    za_grid : 1-D np.array
        np.array for zenith-angle grid [degree]

    aa_grid : 1-D np.array
        np.array for azimuth-angle grid [degree]

    equiv_radius : number
        equivalent volume radius [micrometer]

    NP : integer or None
        code for shape: -1 for spheroid, -2 for cylinder,
        positive for chebyshev, None for arbitrary shape
        (will not use tmatrix for calculations)

    phase : string
        ice, liquid

    aspect_ratio : number
        Aspect ratio [no unit]

    Some inputs have default values, see SingleScatteringData.defaults.
    """

    defaults = {"ptype": PARTICLE_TYPE_MACROS_ISO,
                # as defined in optproperties.h
                "T_grid": np.array([250]),
                "za_grid": np.arange(0, 181, 10),
                "aa_grid": np.arange(0, 181, 10),
                "equiv_radius": 200,  # equivalent volume radius
                "NP": -1,
                # -1 for spheroid, -2 for cylinder, positive for chebyshev
                # set to None for non-tmatrix (this will make calculations
                # impossible, making this object just a data container)
                'phase': 'ice',
                "aspect_ratio": 1.000001}

    mrr = None
    mri = None

    def __init__(self, params={}, **kwargs):
        """See class documentation for constructor info.
        """

        # enable keyword arguments
        if kwargs and not params:
            params = kwargs

        params = general.dict_combine_with_default(params,
                                                   self.__class__.defaults)
        # check parameters
        # make sure grids are np np.arrays
        for grid in ['f', 'T', 'za', 'aa']:
            params[grid + '_grid'] = np.array(params[grid + '_grid'])
        if params['aspect_ratio'] == 1:
            raise ValueError(
                "'aspect_ratio' can not be set to exactly 1 due to numerical difficulties in the T-matrix code. use 1.000001 or 0.999999 instead.")

        if "description" in params:
            self.description = params["description"]
        else:
            self.description = "This arts particle file was generated by the arts_scat Python\n" + \
                               "module, which uses the T-matrix code of Mishchenko to calculate single \n" + \
                               "scattering properties. The parameters used to create this file are shown\n" + \
                               "below\n" + str(params)
        for k, v in params.items():
            setattr(self, k, v)

    def to_xml(self):
        xml_obj = artsXML.XML_Obj('SingleScatteringData')
        xml_obj.write(artsXML.number_to_xml("Index", self.ptype))
        xml_obj.write(
            artsXML.text_to_xml("String",
                                "\"" + general.convert_to_string(
                                    self.description) + "\""))
        xml_obj.write(artsXML.tensor_to_xml(np.array(self.f_grid)))
        xml_obj.write(artsXML.tensor_to_xml(np.array(self.T_grid)))
        xml_obj.write(artsXML.tensor_to_xml(np.array(self.za_grid)))
        xml_obj.write(artsXML.tensor_to_xml(np.array(self.aa_grid)))
        xml_obj.write(artsXML.tensor_to_xml(self.pha_mat_data))
        xml_obj.write(artsXML.tensor_to_xml(self.ext_mat_data))
        xml_obj.write(artsXML.tensor_to_xml(self.abs_vec_data))
        return xml_obj.finalise().str

    @general.force_encoded_string_output
    def __repr__(self):
        S = StringIO()
        S.write("<SingleScatteringData ")
        S.write("ptype=%d " % self.ptype)
        S.write("phase=%s " % self.phase)
        S.write("equiv_radius=%4e" % self.equiv_radius)
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

    def integrate(self):
        """Calculate integrated Z over the sphere.

        Only work for azimuthally symmetrically oriented particles.
        """
        if self.ptype == PARTICLE_TYPE_MACROS_ISO:
            return arts_math.integrate_phasemat(
                self.za_grid,
                self.pha_mat_data[..., 0].squeeze())
        else:
            raise general.PyARTSError("Integration implemented only for"
                                      "ptype = %d. Found ptype = %s" %
                                      (PARTICLE_TYPE_MACROS_ISO, self.ptype))

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
            raise general.PyARTSError("Slicing implemented only for"
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

        raises PyARTSError if not. Otherwise, do nothing.
        """
        if not ((self.f_grid.size or 1, self.T_grid.size or 1) ==
                    self.ext_mat_data.shape[:2] ==
                    self.pha_mat_data.shape[:2] ==
                    self.abs_vec_data.shape[:2] and
                        (self.za_grid.size or 1, self.aa_grid.size or 1) ==
                        self.pha_mat_data.shape[2:4]):
            raise general.PyARTSError(
                "Inconsistent sizes in SingleScatteringData.\n"
                "f_grid: %s, T_grid: %s, za_grid: %s, aa_grid: %s, "
                "ext_mat: %s, pha_mat: %s, abs_vec: %s" %
                (self.f_grid.size or 1, self.T_grid.size or 1,
                 self.za_grid.size or 1, self.aa_grid.size or 1,
                 self.ext_mat_data.shape, self.pha_mat_data.shape,
                 self.abs_vec_data.shape))

    @classmethod
    def _load_from_artsXML_object(cls, artsXML_object):
        """Loads a SingleScatteringData object from an artsXML_object.
        """

        params = {
            "description": artsXML_object['String'][1:-1],
            "ptype": artsXML_object['Index'],
            "f_grid": artsXML_object['Vector'],
            "T_grid": artsXML_object['Vector 0'],
            "za_grid": artsXML_object['Vector 1'],
            "aa_grid": artsXML_object['Vector 2'],
        }

        obj = cls(**params)
        obj.pha_mat_data = artsXML_object['Tensor7']
        obj.ext_mat_data = artsXML_object['Tensor5']
        obj.abs_vec_data = artsXML_object['Tensor5 0']
        return obj

    @classmethod
    def load(cls, filename):
        """Loads a SingleScatteringData object from an existing file.

        Note that this can only import data members that are actually in the file
        - so the scattering properties may not be consistent with the
        *params* data member.
        """
        scat_data = artsXML.load(filename)
        return cls._load_from_artsXML_object(scat_data)


class ScatteringMetaData(ArtsType):
    """Represents a ScatteringMetaData object.

    See online ARTS documentation for object details.
    """

    def __init__(self, description, source, refr_index, mass, diameter_max,
                 diameter_volume_equ, diameter_area_equ_aerodynamical):
        self.description = description
        self.source = source
        self.refr_index = refr_index
        self.mass = mass
        self.diameter_max = diameter_max
        self.diameter_volume_equ = diameter_volume_equ
        self.diameter_area_equ_aerodynamical = diameter_area_equ_aerodynamical

    @general.force_encoded_string_output
    def __repr__(self):
        S = StringIO()
        S.write("<ScatteringMetaData ")
        S.write("description=%s " % self.description)
        S.write("source=%s " % self.source)
        S.write("refr_index=%s " % self.refr_index)
        S.write("mass=%4e " % self.mass)
        S.write("diameter_max=%4e " % self.diameter_max)
        S.write("diameter_volume_equ=%4e " % self.diameter_volume_equ)
        S.write(
            "diameter_area_equ_aerodynamical=%4e " % self.diameter_area_equ_aerodynamical)
        return S.getvalue()

    @classmethod
    def _load_from_artsXML_object(cls, artsXML_object):
        """Loads a SingleScatteringData object from an artsXML_object.
        """

        params = {
            "description": artsXML_object['String'][1:-1],
            "source": artsXML_object['String 0'][1:-1],
            "refr_index": artsXML_object['String 1'][1:-1],
            "mass": artsXML_object['Numeric'],
            "diameter_max": artsXML_object['Numeric 0'],
            "diameter_volume_equ": artsXML_object['Numeric 1'],
            "diameter_area_equ_aerodynamical": artsXML_object['Numeric 2'],
        }

        obj = cls(**params)
        return obj

    @classmethod
    def load(cls, filename):
        """Loads a SingleScatteringData object from an existing file.

        Note that this can only import data members that are actually in the file
        - so the scattering properties may not be consistent with the
        *params* data member.
        """
        scat_data = artsXML.load(filename)
        return cls._load_from_artsXML_object(scat_data)

    def to_xml(self):
        xml_obj = artsXML.XML_Obj('ScatteringMetaData',
                                  attributes={'version': 3})
        xml_obj.write(
            artsXML.text_to_xml("String",
                                "\"" + general.convert_to_string(
                                    self.description) + "\""))
        xml_obj.write(
            artsXML.text_to_xml("String",
                                "\"" + general.convert_to_string(
                                    self.source) + "\""))
        xml_obj.write(
            artsXML.text_to_xml("String",
                                "\"" + general.convert_to_string(
                                    self.refr_index) + "\""))
        xml_obj.write(artsXML.number_to_xml("Numeric", self.mass))
        xml_obj.write(artsXML.number_to_xml("Numeric", self.diameter_max))
        xml_obj.write(
            artsXML.number_to_xml("Numeric", self.diameter_volume_equ))
        xml_obj.write(artsXML.number_to_xml("Numeric",
                                            self.diameter_area_equ_aerodynamical))

        return xml_obj.finalise().str


class ArrayOfSingleScatteringData(ArrayOf):
    """Represents an ArrayOfSingleScatteringData.
    """

    contains = SingleScatteringData

    def normalise(self):
        """Call .normalise() on each SSD
        """

        for elem in self:
            elem.normalise()

    def normalised(self):
        """Return normalised copy of self
        """

        c = copy.deepcopy(self)
        for elem in c:
            elem.normalise()
        return c

    def integrate(self):
        """Return list with all integrated values.
        """

        L = []
        for elem in self:
            L.append(elem.integrate())
        return L

    @property
    def sub(self):
        """Get subslice of each element.
        """

        class Subber(object):
            def __getitem__(self2, *args):
                L = self.__class__()
                for elem in self:
                    L.append(elem.__getitem__(*args))
                return L

        return Subber()

    @classmethod
    def fromData(cls, name, shapestr, sizes, frequencies, angles, data):
        """Load from data.

        Do not call directly, call fromYang, fromHong, etc.
        """

        obj = cls()
        obj.smd = ArrayOfScatteringMetaData()
        for (i, size) in enumerate(sizes):
            # add SSD/SMD, one by one
            obj.append(
                cls.dispatcher_SSD[name](size, shapestr, frequencies, angles,
                                         data[:, i]))
            obj.smd.append(
                cls.dispatcher_SMD[name](size, shapestr, frequencies, angles,
                                         data[:, i]))
        return obj


class ArrayOfScatteringMetaData(ArrayOf):
    contains = ScatteringMetaData
