# -*- coding: utf-8 -*-
"""
Implementation of classes to handle various catalogue information.

"""

from .griddedfield import *
from .utils import return_if_arts_type

import numpy as np

__all__ = ['ArrayOfLineRecord',
           'CIARecord',
           'GasAbsLookup',
           'LineMixingRecord',
           'QuantumIdentifier',
           'QuantumNumberRecord',
           'QuantumNumbers',
           'Sparse',
           'SpeciesAuxData',
           'SpeciesTag',
           ]


class ArrayOfLineRecord:
    """Represents an ArrayOfLineRecord object.

    See online ARTS documentation for object details.

    """

    def __init__(self, data=None, version=None):
        self.data = data
        self.version = version

    @property
    def version(self):
        """ArrayOfRecord version number."""
        return self._version

    @property
    def data(self):
        """List of strings representing line records."""
        return self._data

    @version.setter
    def version(self, version):
        if version is None:
            self._version = None
            return

        if type(version) is str:
            self._version = version
        else:
            raise TypeError('version has to be String.')

    @data.setter
    def data(self, data):
        self._data = return_if_arts_type(data, 'ArrayOfString')

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads an ArrayOfLineRecord object from an existing file.
        """
        obj = cls()
        obj.version = xmlelement.attrib['version']
        obj.data = xmlelement.text.strip().split('\n')
        return obj

    def write_xml(self, xmlwriter, attr=None):
        """Write an ArrayOfLineRecord object to an ARTS XML file.
        """
        if attr is None:
            attr = {}

        attr['version'] = self.version
        attr['nelem'] = len(self.data)

        xmlwriter.open_tag("ArrayOfLineRecord", attr)
        xmlwriter.write('\n'.join(self.data) + '\n')
        xmlwriter.close_tag()


class CIARecord:
    """Represents a CIARecord object.

    See online ARTS documentation for object details.

    """

    def __init__(self, molecule1=None, molecule2=None, data=None):
        self.molecule1 = molecule1
        self.molecule2 = molecule2
        self.data = data

    @property
    def molecule1(self):
        """Name of the first molecule."""
        return self._molecule1

    @property
    def molecule2(self):
        """Name of the second molecule."""
        return self._molecule2

    @property
    def data(self):
        """Actual data stored in (list of) GriddedField2 objects."""
        return self._data

    @molecule1.setter
    def molecule1(self, molecule1):
        self._molecule1 = return_if_arts_type(molecule1, 'String')

    @molecule2.setter
    def molecule2(self, molecule2):
        self._molecule2 = return_if_arts_type(molecule2, 'String')

    @data.setter
    def data(self, data):
        self._data = return_if_arts_type(data, 'ArrayOfGriddedField2')

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a CIARecord object from an existing file.
        """

        obj = cls()
        obj.molecule1 = xmlelement.attrib['molecule1']
        obj.molecule2 = xmlelement.attrib['molecule2']
        obj.data = xmlelement[0].value()

        return obj

    def write_xml(self, xmlwriter, attr=None):
        """Write a CIARecord object to an ARTS XML file.
        """
        if attr is None:
            attr = {}

        attr['molecule1'] = self.molecule1
        attr['molecule2'] = self.molecule2

        xmlwriter.open_tag("CIARecord", attr)
        xmlwriter.write_xml(self.data)
        xmlwriter.close_tag()


# TODO(LKL): consider splitting SpeciesAuxData into seperate classes for each
# version. SpeciesAuxData could be used as wrapper class.
class SpeciesAuxData:
    """Represents a SpeciesAuxData object.

    See online ARTS documentation for object details.

    """

    def __init__(self, data, version, nparam=None):
        self.version = version
        self.nparam = nparam
        self.data = data

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a SpeciesAuxData object from an existing file.
        """

        version = int(xmlelement.attrib['version'])

        if version == 1:
            nparam = int(xmlelement.attrib['nparam'])
            data = [s for s in  xmlelement.text.split('\n') if s != '']
        elif version == 2:
            nparam = None
            data = []
            sub_list = []
            for n, elem in enumerate(xmlelement):
                if n != 0 and n % 3 == 0:
                    data.append(sub_list)
                    sub_list = []
                sub_list.append(elem.value())
            data.append(sub_list)

        obj = cls(data, version, nparam=nparam)
        return obj

    def write_xml(self, xmlwriter, attr=None):
        """Write a ScatterinMetaData object to an ARTS XML file.
        """
        if attr is None:
            attr = {}

        attr['version'] = self.version
        attr['nelem'] = len(self.data)

        if self.version == 1:
            attr['nparam'] = self.nparam

            xmlwriter.open_tag("SpeciesAuxData", attr)
            xmlwriter.write('\n'.join(self.data) + '\n')
            xmlwriter.close_tag()
        elif self.version == 2:
            xmlwriter.open_tag("SpeciesAuxData", attr)
            for sub_list in self.data:
                for element in sub_list:
                    xmlwriter.write_xml(element)
            xmlwriter.close_tag()


class GasAbsLookup:
    """Represents a GasAbsLookup object.

    See online ARTS documentation for object details.

    """

    def __init__(self, speciestags=None, nonlinearspecies=None,
            frequencygrid=None, pressuregrid=None, referencevmrprofiles=None,
            referencetemperatureprofile=None, temperaturepertubations=None,
            nonlinearspeciesvmrpertubations=None, absorptioncrosssection=None):

        self.speciestags = speciestags
        self.nonlinearspecies = nonlinearspecies
        self.frequencygrid = frequencygrid
        self.pressuregrid = pressuregrid
        self.referencevmrprofiles = referencevmrprofiles
        self.referencetemperatureprofile = referencetemperatureprofile
        self.temperaturepertubations = temperaturepertubations
        self.nonlinearspeciesvmrpertubations = nonlinearspeciesvmrpertubations
        self.absorptioncrosssection = absorptioncrosssection

    @property
    def speciestags(self):
        """List of :class:`SpeciesTag`."""
        return self._speciestags

    @property
    def nonlinearspecies(self):
        """Indices to indentify nonlinear species."""
        return self._nonlinearspecies

    @property
    def frequencygrid(self):
        """Frequency vector."""
        return self._frequencygrid

    @property
    def pressuregrid(self):
        """Pressure level vector."""
        return self._pressuregrid

    @property
    def referencevmrprofiles(self):
        """Reference VMR profiles."""
        return self._referencevmrprofiles

    @property
    def referencetemperatureprofile(self):
        """Reference temperature profile."""
        return self._referencetemperatureprofile

    @property
    def temperaturepertubations(self):
        """Vector with temperature pertubations."""
        return self._temperaturepertubations

    @property
    def nonlinearspeciesvmrpertubations(self):
        """Vector with VMR pertubations for nonlinear species."""
        return self._nonlinearspeciesvmrpertubations

    @property
    def absorptioncrosssection(self):
        """Absorption crosssections."""
        return self._absorptioncrosssection

    @speciestags.setter
    def speciestags(self, speciestags):
        self._speciestags = return_if_arts_type(
                speciestags, 'ArrayOfArrayOfSpeciesTag')

    @nonlinearspecies.setter
    def nonlinearspecies(self, nonlinearspecies):
        self._nonlinearspecies = return_if_arts_type(
                nonlinearspecies, 'ArrayOfIndex')

    @frequencygrid.setter
    def frequencygrid(self, frequencygrid):
        self._frequencygrid = return_if_arts_type(
                frequencygrid, 'Vector')

    @pressuregrid.setter
    def pressuregrid(self, pressuregrid):
        self._pressuregrid = return_if_arts_type(
                pressuregrid, 'Vector')

    @referencevmrprofiles.setter
    def referencevmrprofiles(self, referencevmrprofiles):
        self._referencevmrprofiles = return_if_arts_type(
                referencevmrprofiles, 'Matrix')

    @referencetemperatureprofile.setter
    def referencetemperatureprofile(self, referencetemperatureprofile):
        self._referencetemperatureprofile = return_if_arts_type(
                referencetemperatureprofile, 'Vector')

    @temperaturepertubations.setter
    def temperaturepertubations(self, temperaturepertubations):
        self._temperaturepertubations = return_if_arts_type(
                temperaturepertubations, 'Vector')

    @nonlinearspeciesvmrpertubations.setter
    def nonlinearspeciesvmrpertubations(self, nonlinearspeciesvmrpertubations):
        self._nonlinearspeciesvmrpertubations = return_if_arts_type(
                nonlinearspeciesvmrpertubations, 'Vector')

    @absorptioncrosssection.setter
    def absorptioncrosssection(self, absorptioncrosssection):
        self._absorptioncrosssection = return_if_arts_type(
                absorptioncrosssection, 'Tensor4')

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a GasAbsLookup object from an existing file.
        """

        obj = cls()
        obj.speciestags = xmlelement[0].value()
        obj.nonlinearspecies = xmlelement[1].value()
        obj.frequencygrid = xmlelement[2].value()
        obj.pressuregrid = xmlelement[3].value()
        obj.referencevmrprofiles = xmlelement[4].value()
        obj.referencetemperatureprofile = xmlelement[5].value()
        obj.temperaturepertubations = xmlelement[6].value()
        obj.nonlinearspeciesvmrpertubations = xmlelement[7].value()
        obj.absorptioncrosssection = xmlelement[8].value()

        return obj

    def write_xml(self, xmlwriter, attr=None):
        """Write a ScatterinMetaData object to an ARTS XML file.
        """
        if attr is None:
            attr = {}

        xmlwriter.open_tag("GasAbsLookup", attr)
        xmlwriter.write_xml(self.speciestags)
        xmlwriter.write_xml(self.nonlinearspecies,
                            {'name': 'NonlinearSpecies'},
                            arraytype='Index')
        xmlwriter.write_xml(self.frequencygrid,
                            {'name': 'FrequencyGrid'})
        xmlwriter.write_xml(self.pressuregrid,
                            {'name': 'PressureGrid'})
        xmlwriter.write_xml(self.referencevmrprofiles,
                            {'name': 'ReferenceVmrProfiles'})
        xmlwriter.write_xml(self.referencetemperatureprofile,
                            {'name': 'ReferenceTemperatureProfile'})
        xmlwriter.write_xml(self.temperaturepertubations,
                            {'name': 'TemperaturePertubations'})
        xmlwriter.write_xml(self.nonlinearspeciesvmrpertubations,
                            {'name': 'NonlinearSpeciesVmrPertubations'})
        xmlwriter.write_xml(self.absorptioncrosssection,
                            {'name': 'AbsorptionsCrossSections'})
        xmlwriter.close_tag()


class SpeciesTag(str):
    """Represents a SpeciesTag object.

    See online ARTS documentation for object details.

    """

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a SpeciesTag object from an existing file.
        """
        if xmlelement.text is None:
            raise Exception('SpeciesTag must not be empty.')
        return cls(xmlelement.text.strip()[1:-1])

    def write_xml(self, xmlwriter, attr=None):
        """Write a SpeciesTag object to an ARTS XML file.
        """
        if attr is None:
            attr = {}

        xmlwriter.open_tag('SpeciesTag', attr, newline=False)
        xmlwriter.write('"' + self + '"')
        xmlwriter.close_tag()


class Sparse():
    """Represents a Sparse object.

    See online ARTS documentation for object details.

    """

    def __init__(self, nrows=None, ncols=None, rowindex=None, colindex=None,
            sparsedata=None):

        self.nrows = nrows
        self.ncols = ncols
        self.rowindex = rowindex
        self.colindex = colindex
        self.sparsedata = sparsedata

    @property
    def nrows(self):
        """Number of rows."""
        return self._nrows

    @property
    def ncols(self):
        """Number of columns."""
        return self._ncols

    @property
    def rowindex(self):
        """Row indices to locate data in matrix."""
        return self._rowindex

    @property
    def colindex(self):
        """Column indices to locate data in matrix."""
        return self._colindex

    @property
    def sparsedata(self):
        """Data value at specified positions in matrix."""
        return self._sparsedata

    @nrows.setter
    def nrows(self, nrows):
        self._nrows = return_if_arts_type(nrows, 'Index')

    @ncols.setter
    def ncols(self, ncols):
        self._ncols = return_if_arts_type(ncols, 'Index')

    @rowindex.setter
    def rowindex(self, rowindex):
        self._rowindex = return_if_arts_type(rowindex, 'Vector')

    @colindex.setter
    def colindex(self, colindex):
        self._colindex = return_if_arts_type(colindex, 'Vector')

    @sparsedata.setter
    def sparsedata(self, sparsedata):
        self._sparsedata = return_if_arts_type(sparsedata, 'Vector')

    def to_csc_matrix(self):
        """ Returns a scipy sparse object """
        from scipy.sparse import csc_matrix

        self.check_dimension()
        obj = csc_matrix((self.sparsedata, (self.rowindex, self.colindex)),
                [self.nrows, self.ncols])

        return obj

    def check_dimension(self):
        """Checks the consistency of stored data.

        Note:
            This check is done automatically before storing and after loading
            XML files.

        """
        if self.rowindex.size == self.colindex.size == self.sparsedata.size:
            return True
        else:
            raise Exception(
                'RowIndex, ColIndex and SparseData must have same length.')

    @classmethod
    def from_csc_matrix(cls, csc_matrix):
        """Creates a Sparse object from a scipy sparse object.

        Parameters:
            cscs_matrix (sp.csc_matrix): scipy sparse object.

        Returns:
            Sparse: typhon Sparse object.

        """

        obj = cls()
        obj.nrows = csc_matrix.shape[0]
        obj.ncols = csc_matrix.shape[1]
        csc = csc_matrix.tocoo()
        obj.rowindex = csc.row
        obj.colindex = csc.col
        obj.sparsedata = csc_matrix.data
        obj.check_dimension()

        return obj

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a Sparse object from an existing file.
        """

        binaryfp = xmlelement.binaryfp
        nelem = int(xmlelement[0].attrib['nelem'])
        obj = cls()
        obj.nrows = int(xmlelement.attrib['nrows'])
        obj.ncols = int(xmlelement.attrib['ncols'])

        if binaryfp is None:
            obj.rowindex = np.fromstring(xmlelement[0].text, sep=' ').astype(int)
            obj.colindex = np.fromstring(xmlelement[1].text, sep=' ').astype(int)
            obj.sparsedata = np.fromstring(xmlelement[2].text, sep=' ')
        else:
            obj.rowindex = np.fromfile(binaryfp, dtype='<i4', count=nelem)
            obj.colindex = np.fromfile(binaryfp, dtype='<i4', count=nelem)
            obj.sparsedata = np.fromfile(binaryfp, dtype='<d', count=nelem)

        obj.check_dimension()

        return obj

    def write_xml(self, xmlwriter, attr=None):
        """Write a Sparse object to an ARTS XML file.
        """

        self.check_dimension()

        precision = xmlwriter.precision

        if attr is None:
            attr = {}

        attr['nrows'] = self.nrows
        attr['ncols'] = self.ncols

        xmlwriter.open_tag('Sparse', attr)

        binaryfp = xmlwriter.binaryfilepointer

        if binaryfp is None:
            xmlwriter.open_tag('RowIndex', {'nelem': self.rowindex.size})
            for i in self.rowindex:
                xmlwriter.write('%d' % i + '\n')
            xmlwriter.close_tag()
            xmlwriter.open_tag('ColIndex', {'nelem': self.colindex.size})
            for i in self.colindex:
                xmlwriter.write('%d' % i + '\n')
            xmlwriter.close_tag()
            xmlwriter.open_tag('SparseData', {'nelem': self.sparsedata.size})
            for i in self.sparsedata:
                xmlwriter.write(('%' + precision) % i + '\n')
            xmlwriter.close_tag()
            xmlwriter.close_tag()
        else:
            xmlwriter.open_tag('RowIndex', {'nelem': self.rowindex.size})
            np.array(self.rowindex, dtype='i4').tofile(binaryfp)
            xmlwriter.close_tag()
            xmlwriter.open_tag('ColIndex', {'nelem': self.colindex.size})
            np.array(self.colindex, dtype='i4').tofile(binaryfp)
            xmlwriter.close_tag()
            xmlwriter.open_tag('SparseData', {'nelem': self.sparsedata.size})
            np.array(self.sparsedata, dtype='d').tofile(binaryfp)
            xmlwriter.close_tag()
            xmlwriter.close_tag()


class QuantumIdentifier(str):
    """Represents a QuantumIdentifier object.

    See online ARTS documentation for object details.

    """

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a QuantumIdentifier object from an existing file.
        """
        if xmlelement.text is None:
            raise Exception('QuantumIdentifier must not be empty.')
        return cls(xmlelement.text.strip())

    def write_xml(self, xmlwriter, attr=None):
        """Write a QuantumIdentifier object to an ARTS XML file.
        """
        if attr is None:
            attr = {}

        xmlwriter.open_tag('QuantumIdentifier', attr, newline=False)
        xmlwriter.write(self)
        xmlwriter.close_tag()


class QuantumNumberRecord():
    """Represents a QuantumNumberRecord object.

    See online ARTS documentation for object details.

    """

    def __init__(self, upper=None, lower=None):

        self.lower = lower
        self.upper = upper

    @property
    def upper(self):
        """QuantumNumbers object representing the upper quantumnumber."""
        return self._upper

    @property
    def lower(self):
        """QuantumNumbers object representing the lower quantumnumber."""
        return self._lower

    @upper.setter
    def upper(self, upper):
        self._upper = return_if_arts_type(upper, 'QuantumNumbers')

    @lower.setter
    def lower(self, lower):
        self._lower = return_if_arts_type(lower, 'QuantumNumbers')

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a QuantumNumberRecord object from an existing file.
        """

        obj = cls()
        obj.upper = xmlelement[0][0].value()
        obj.lower = xmlelement[1][0].value()

        return obj

    def write_xml(self, xmlwriter, attr=None):
        """Write a SpeciesTag object to an ARTS XML file.
        """
        if attr is None:
            attr = {}

        xmlwriter.open_tag('QuantumNumberRecord', attr)
        xmlwriter.open_tag('Upper', attr, newline=False)
        xmlwriter.write_xml(self.upper)
        xmlwriter.close_tag()
        xmlwriter.open_tag('Lower', attr, newline=False)
        xmlwriter.write_xml(self.lower)
        xmlwriter.close_tag()
        xmlwriter.close_tag()


class QuantumNumbers():
    """Represents a QuantumNumbers object.

    See online ARTS documentation for object details.

    """

    def __init__(self, numbers=None, nelem=None):

        self.numbers = numbers
        self.nelem = nelem

    @property
    def numbers(self):
        """String representing the quantumnumbers."""
        return self._numbers

    @property
    def nelem(self):
        """Number of quantumnumbers stored."""
        return self._nelem

    @numbers.setter
    def numbers(self, numbers):
        self._numbers = return_if_arts_type(numbers, 'String')

    @nelem.setter
    def nelem(self, nelem):
        if nelem is None:
            self._nelem = None
            return

        self._nelem = nelem

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a QuantumNumbers object from an existing file.
        """

        obj = cls()
        obj.numbers = xmlelement.text
        obj.nelem = int(xmlelement.attrib['nelem'])

        return obj


    def write_xml(self, xmlwriter, attr=None):
        """Write a SpeciesTag object to an ARTS XML file.
        """
        if attr is None:
            attr = {}

        attr['nelem'] = self.nelem

        xmlwriter.open_tag('QuantumNumbers', attr, newline=False)
        xmlwriter.write(self.numbers)
        xmlwriter.close_tag(newline=False)


class LineMixingRecord():
    """Represents a LineMixingRecord object.

    See online ARTS documentation for object details.

    """

    def __init__(self, tag=None, quantumnumberrecord=None, data=None):

        self.tag = tag
        self.quantumnumberrecord = quantumnumberrecord
        self.data = data

    @property
    def tag(self):
        """:class:`SpeciesTag`"""
        return self._tag

    @property
    def quantumnumberrecord(self):
        """:class:`QuantumNumberRecord`"""
        return self._quantumnumberrecord

    @property
    def data(self):
        """Lineshape parameters."""
        return self._data

    @tag.setter
    def tag(self, tag):
        if tag is None:
            self._tag = None
            return

        self._tag = SpeciesTag(tag)

    @quantumnumberrecord.setter
    def quantumnumberrecord(self, quantumnumberrecord):
        self._quantumnumberrecord = return_if_arts_type(
                quantumnumberrecord, 'QuantumNumberRecord')

    @data.setter
    def data(self, data):
        self._data = return_if_arts_type( data, 'Vector')

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a LineMixingRecord object from an existing file.
        """

        obj = cls()
        obj.tag = xmlelement[0].value()
        obj.quantumnumberrecord = xmlelement[1].value()
        obj.data = xmlelement[2].value()

        return obj

    def write_xml(self, xmlwriter, attr=None):
        """Write a LineMixingRecord object to an ARTS XML file.
        """
        if attr is None:
            attr = {}

        xmlwriter.open_tag("LineMixingRecord", attr)
        xmlwriter.write_xml(self.tag)
        xmlwriter.write_xml(self.quantumnumberrecord)
        xmlwriter.write_xml(self.data)
        xmlwriter.close_tag()
