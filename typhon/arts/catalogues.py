# -*- coding: utf-8 -*-
"""
Implementation of classes to handle various catalogue information.

"""

import numpy as np

__all__ = ['ArrayOfLineRecord',
           'CIARecord',
           'SpeciesAuxData',
           'GasAbsLookup',
           'SpeciesTag',
           'Sparse',
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

        self._version = version

    @data.setter
    def data(self, data):
        if data is None:
            self._data = None
            return

        if type(data) is list and type(data[0]) is str:
            self._data = data
        else:
            raise TypeError('data has to be a list of strings.')

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
        """Actual data stored in (list of) GriddedField2 objcts."""
        return self._data

    @molecule1.setter
    def molecule1(self, molecule1):
        if molecule1 is None:
            self._molecule1 = None
            return

        if type(molecule1) is str:
            self._molecule1 = molecule1
        else:
            raise TypeError('molecule1 has to be str.')

    @molecule2.setter
    def molecule2(self, molecule2):
        if molecule2 is None:
            self._molecule2 = None
            return

        if type(molecule2) is str:
            self._molecule2 = molecule2
        else:
            raise TypeError('molecule2 has to be str.')

    @data.setter
    def data(self, data):
        if data is None:
            self._data = None
            return

        self._data = data

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
        """List of SpeciesTags."""
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
        if speciestags is None:
            self._speciestags = None
            return

        if not type(speciestags) is list:
            raise TypeError('speciestags has to be a list.')

        if not (type(speciestags[0]) is list
            and type(speciestags[0][0]) is SpeciesTag):
            raise TypeError('speciestags entries have to be a ArrayOfSpeciesTag.')

        self._speciestags = speciestags

    @nonlinearspecies.setter
    def nonlinearspecies(self, nonlinearspecies):
        if nonlinearspecies is None:
            self._nonlinearspecies = None
            return

        self._nonlinearspecies = nonlinearspecies

    @frequencygrid.setter
    def frequencygrid(self, frequencygrid):
        if frequencygrid is None:
            self._frequencygrid = None
            return

        if type(frequencygrid) is np.ndarray:
            self._frequencygrid = frequencygrid
        else:
            raise TypeError('frequencygrid has to be np.ndarray.')

    @pressuregrid.setter
    def pressuregrid(self, pressuregrid):
        if pressuregrid is None:
            self._pressuregrid = None
            return

        if type(pressuregrid) is np.ndarray:
            self._pressuregrid = pressuregrid
        else:
            raise TypeError('pressuregrid has to be np.ndarray.')

    @referencevmrprofiles.setter
    def referencevmrprofiles(self, referencevmrprofiles):
        if referencevmrprofiles is None:
            self._referencevmrprofiles = None
            return

        if type(referencevmrprofiles) is np.ndarray:
            self._referencevmrprofiles = referencevmrprofiles
        else:
            raise TypeError('referencevmrprofiles has to be np.ndarray.')

    @referencetemperatureprofile.setter
    def referencetemperatureprofile(self, referencetemperatureprofile):
        if referencetemperatureprofile is None:
            self._referencetemperatureprofile = None
            return

        if type(referencetemperatureprofile) is np.ndarray:
            self._referencetemperatureprofile = referencetemperatureprofile
        else:
            raise TypeError('referencetemperatureprofile has to be np.ndarray.')

    @temperaturepertubations.setter
    def temperaturepertubations(self, temperaturepertubations):
        if temperaturepertubations is None:
            self._temperaturepertubations = None
            return

        if type(temperaturepertubations) is np.ndarray:
            self._temperaturepertubations = temperaturepertubations
        else:
            raise TypeError('temperaturepertubations has to be np.ndarray.')

    @nonlinearspeciesvmrpertubations.setter
    def nonlinearspeciesvmrpertubations(self, nonlinearspeciesvmrpertubations):
        if nonlinearspeciesvmrpertubations is None:
            self._nonlinearspeciesvmrpertubations = None
            return

        if type(nonlinearspeciesvmrpertubations) is np.ndarray:
            self._nonlinearspeciesvmrpertubations = nonlinearspeciesvmrpertubations
        else:
            raise TypeError('nonlinearspeciesvmrpertubations has to be np.ndarray.')

    @absorptioncrosssection.setter
    def absorptioncrosssection(self, absorptioncrosssection):
        if absorptioncrosssection is None:
            self._absorptioncrosssection = None
            return

        self._absorptioncrosssection = absorptioncrosssection
        if type(absorptioncrosssection) is np.ndarray:
            self._absorptioncrosssection = absorptioncrosssection
        else:
            raise TypeError('absorptioncrosssection has to be np.ndarray.')

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
                            {'name': 'NonlinearSpecies'})
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
        if nrows is None:
            self._nrows = None
            return

        self._nrows = nrows

    @ncols.setter
    def ncols(self, ncols):
        if ncols is None:
            self._ncols = None
            return

        self._ncols = ncols

    @rowindex.setter
    def rowindex(self, rowindex):
        if rowindex is None:
            self._rowindex = None
            return

        if type(rowindex) is np.ndarray:
            self._rowindex = rowindex
        else:
            raise TypeError('rowindex has to be np.ndarray.')

    @colindex.setter
    def colindex(self, colindex):
        if colindex is None:
            self._colindex = None
            return

        if type(colindex) is np.ndarray:
            self._colindex = colindex
        else:
            raise TypeError('colindex has to be np.ndarray.')

    @sparsedata.setter
    def sparsedata(self, sparsedata):
        if sparsedata is None:
            self._sparsedata = None
            return

        if type(sparsedata) is np.ndarray:
            self._sparsedata = sparsedata
        else:
            raise TypeError('sparsedata has to be np.ndarray.')

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

        obj = cls()
        obj.nrows = int(xmlelement.attrib['nrows'])
        obj.ncols = int(xmlelement.attrib['ncols'])
        obj.rowindex = np.fromstring(xmlelement[0].text, sep=' ').astype(int)
        obj.colindex = np.fromstring(xmlelement[1].text, sep=' ').astype(int)
        obj.sparsedata = np.fromstring(xmlelement[2].text, sep=' ')
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

