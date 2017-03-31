# -*- coding: utf-8 -*-
"""
Implementation of classes to handle various catalogue information.

"""

try:
    from .utils import return_if_arts_type
except:
    from typhon.arts.utils import return_if_arts_type


import numpy as np
import scipy.sparse
from fractions import Fraction as _R

__all__ = ['ArrayOfLineRecord',
           'ARTSCAT5',
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

    def __repr__(self):
        return "ArrayOfLineRecord. " + self.version + ". " + \
            str(len(self.data)) + " lines."

    def __getitem__(self, index):
        return self.data[index]

    def as_ARTSCAT5(self):
        """Returns manipulable ARTSCAT5 class of this linerecord array
        """
        assert self.version == 'ARTSCAT-5', "Only for ARTSCAT-5 data"
        return ARTSCAT5(linerecord_array=self)

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

        if isinstance(version, str):
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


class ARTSCAT5:
    """Class to contain ARTSCAT entries that can be accessed and  manipulated

        Access this data as
        (N, I, F0, S0, T0, E0, A, GL, GU, PB, QN, LM)  = ARTSCAT5[line_nr],
        where N is the name of the species, I is the AFGL isotopological code,
        F0 is the central frequency, S0 is the line strength at temperature T0,
        E0 is the lower energy state, A is the einstein coefficient, GL is the
        lower population constant, GU is the upper population constant, PB
        is a dictionary of the pressurebroadening scheme, QN is a
        QuantumNumberRecord, and LM is a line-mixing dictionary.  The
        dictionaries have keys corresponding to the ARTSCAT tags.  line_nr is
        an index absolutely less than len(self)

        Note:  Must be ARTSCAT5 line type or this will leave the class data
        in disarray.

        Future tech debt 1: The reading of tagged data will fail if major tags
        ever match minor tags (ex. PB is major, N2 is minor for PB, if LM ever
        gets the PB minor tag, then the method below will fail).

        Future tech debt 2: To add version number and make the class general,
        ARTSCAT3 only have minor tag N2 for line mixing and ARTSCAT4 has AP.
        These tags are however implicit and not written.  A
    """

    _spec_ind = 0
    _iso_ind = 1
    _freq_ind = 2
    _str_ind = 3
    _t0_ind = 4
    _elow_ind = 5
    _ein_ind = 6
    _glow_ind = 7
    _gupp_ind = 8
    _pb_ind = 9
    _qn_ind = 10
    _lm_ind = 11

    def __init__(self, linerecord_array=None):
        self._dictionaries = np.array([], dtype=dict)
        self._n = 0
        self.LineRecordData = {
                'freq': np.array([]),
                'afgl': np.array([], dtype='int'),
                'str': np.array([]),
                'glow': np.array([]),
                'gupp': np.array([]),
                'elow': np.array([]),
                'spec': np.array([], dtype='str'),
                'ein': np.array([]),
                't0': np.array([])}

        if linerecord_array is None:
            return

        if linerecord_array.version != 'ARTSCAT-5':
            raise RuntimeError("linerecord_array not version 5")

        for linerecord_str in linerecord_array:
            self.append_linestr(linerecord_str)
        self._assert_sanity_()

    def append_linestr(self, linerecord_str):
        """Takes an arts-xml catalog string and appends info to the class data
        """
        lr = linerecord_str.split()
        len_lr = len(lr)
        self._dictionaries = np.append(self._dictionaries,
                                       {"QN": QuantumNumberRecord(),
                                        "PB": {"Type": None,
                                               "Data": np.array([])},
                                        "LM": {"Type": None,
                                               "Data": np.array([])}})

        spec = lr[1].split('-')
        self.LineRecordData['spec'] = np.append(self.LineRecordData['spec'],
                                                spec[self._spec_ind])
        self.LineRecordData['afgl'] = np.append(self.LineRecordData['afgl'],
                                                int(spec[self._iso_ind]))
        self.LineRecordData['freq'] = np.append(self.LineRecordData['freq'],
                                                float(lr[self._freq_ind]))
        self.LineRecordData['str'] = np.append(self.LineRecordData['str'],
                                               float(lr[self._str_ind]))
        self.LineRecordData['t0'] = np.append(self.LineRecordData['t0'],
                                              float(lr[self._t0_ind]))
        self.LineRecordData['elow'] = np.append(self.LineRecordData['elow'],
                                                float(lr[self._elow_ind]))
        self.LineRecordData['ein'] = np.append(self.LineRecordData['ein'],
                                               float(lr[self._ein_ind]))
        self.LineRecordData['glow'] = np.append(self.LineRecordData['glow'],
                                                float(lr[self._glow_ind]))
        self.LineRecordData['gupp'] = np.append(self.LineRecordData['gupp'],
                                                float(lr[self._gupp_ind]))
        self._n += 1

        key = lr[9]
        i = 10
        qnr = ''
        while i < len_lr:
            this = lr[i]
            if this in ['QN', 'PB', 'LM']:
                key = this
            elif key == 'QN':
                qnr += ' ' + this
            else:
                try:
                    self._dictionaries[-1][key]["Data"] = \
                        np.append(self._dictionaries[-1][key]["Data"],
                                  float(this))
                except:
                    self._dictionaries[-1][key]["Type"] = this
            i += 1
        self._dictionaries[-1]['QN'] = QuantumNumberRecord.from_str(qnr)

    def append_line(self, line):
        """Appends a line from data
        """
        self.LineRecordData['spec'] = np.append(self.LineRecordData['spec'],
                                                str(line[self._spec_ind]))
        self.LineRecordData['afgl'] = np.append(self.LineRecordData['afgl'],
                                                int(line[self._iso_ind]))
        self.LineRecordData['freq'] = np.append(self.LineRecordData['freq'],
                                                line[self._freq_ind])
        self.LineRecordData['str'] = np.append(self.LineRecordData['str'],
                                               line[self._str_ind])
        self.LineRecordData['t0'] = np.append(self.LineRecordData['t0'],
                                              line[self._t0_ind])
        self.LineRecordData['elow'] = np.append(self.LineRecordData['elow'],
                                                line[self._elow_ind])
        self.LineRecordData['ein'] = np.append(self.LineRecordData['ein'],
                                               line[self._ein_ind])
        self.LineRecordData['glow'] = np.append(self.LineRecordData['glow'],
                                                line[self._glow_ind])
        self.LineRecordData['gupp'] = np.append(self.LineRecordData['gupp'],
                                                line[self._gupp_ind])
        self._dictionaries = np.append(self._dictionaries,
                                       {'PB': line[self._pb_ind],
                                        'QN': line[self._qn_ind],
                                        'LM': line[self._lm_ind]})
        self._n += 1

    def append_ArrayOfLineRecord(self, array_of_linerecord, sort=True):
        """Appends lines in ArrayOfLineRecord to ARTSCAT5
        """
        assert array_of_linerecord.version == 'ARTSCAT-5', "Only for ARTSCAT-5"
        for l in array_of_linerecord:
            self.append_linestr(l)
        self._assert_sanity_()
        self.sort()

    def append_ARTSCAT5(self, artscat5, sort=True):
        """Appends all the lines of another artscat5 to this
        """
        for line in artscat5:
            self.append_line(line)
        self._assert_sanity_()
        if sort:
            self.sort()

    def sort(self, kind='freq', ascending=True):
        """Sorts the ARTSCAT5 data by kind.  Set ascending to False for
        descending sorting
        """
        i = np.argsort(self.LineRecordData[kind])
        if not ascending:
            i = i[::-1]

        for key in self.LineRecordData:
            self.LineRecordData[key] = self.LineRecordData[key][i]
        self._dictionaries = self._dictionaries[i]

    def remove(self, upper_limit=None, lower_limit=None, kind='freq'):
        """Removes lines not within limits of kind

        examples: remove(upper_limit=1e12) removes all lines with frequency
        above 1 THz. remove(lower_limit=1e-27, kind='str') removes all lines
        with line strength below 1e-27
        """
        assert upper_limit is not None or lower_limit is not None, \
            "Cannot remove lines when the limits are undeclared"
        remove_these = []
        for i in range(self._n):
            if lower_limit is not None:
                if self.LineRecordData[kind][i] < lower_limit:
                    remove_these.append(i)
            elif upper_limit is not None:
                if self.LineRecordData[kind][i] > upper_limit:
                    remove_these.append(i)

        for i in remove_these[::-1]:
            self.remove_line(i)

    def __repr__(self):
        return "ARTSCAT-5 with " + str(self._n) + " lines. Species: " + \
            str(np.unique(self.LineRecordData['spec']))

    def __len__(self):
        return self._n

    def _assert_sanity_(self):
        """Helper to assert that the data is good
        """
        assert self._n == len(self.LineRecordData['freq']) and \
            self._n == len(self.LineRecordData['spec']) and    \
            self._n == len(self.LineRecordData['afgl']) and    \
            self._n == len(self.LineRecordData['str']) and     \
            self._n == len(self.LineRecordData['t0']) and      \
            self._n == len(self.LineRecordData['elow']) and    \
            self._n == len(self.LineRecordData['ein']) and     \
            self._n == len(self.LineRecordData['glow']) and    \
            self._n == len(self.LineRecordData['gupp']) and    \
            self._n == len(self._dictionaries),                \
            self._error_in_length_message_()

    def __getitem__(self, index):
        """Returns a single line as tuple --- TODO: create LineRecord class?
        """
        assert abs(index) < self._n, "Out of bounds"
        return (self.LineRecordData['spec'][index],
                self.LineRecordData['afgl'][index],
                self.LineRecordData['freq'][index],
                self.LineRecordData['str'][index],
                self.LineRecordData['t0'][index],
                self.LineRecordData['elow'][index],
                self.LineRecordData['ein'][index],
                self.LineRecordData['glow'][index],
                self.LineRecordData['gupp'][index],
                self.pressurebroadening(index),
                self.quantumnumbers(index),
                self.linemixing(index))

    def get_arts_str(self, index):
        """Returns the arts-xml catalog string for line at index
        """
        l = self[index]
        s = '@ ' + l[self._spec_ind] + '-' + str(l[self._iso_ind])
        s += ' ' + str(l[self._freq_ind])
        s += ' ' + str(l[self._str_ind])
        s += ' ' + str(l[self._t0_ind])
        s += ' ' + str(l[self._elow_ind])
        s += ' ' + str(l[self._ein_ind])
        s += ' ' + str(l[self._glow_ind])
        s += ' ' + str(l[self._gupp_ind])
        if l[self._pb_ind]['Type'] is not None:
            s += ' PB ' + l[self._pb_ind]['Type']
            for p in l[self._pb_ind]['Data']:
                s += ' ' + str(p)
        s += str(self.quantumnumbers(index))
        if l[self._lm_ind]['Type'] is not None:
            s += 'LM ' + l[self._lm_ind]['Type']
            for p in l[self._lm_ind]['Data']:
                s += ' ' + str(p)
        return s

    def pressurebroadening(self, index):
        """Return pressure broadening entries for line at index
        """
        return self._dictionaries[index]['PB']

    def quantumnumbers(self, index):
        """Return quantum number entries for line at index
        """
        return self._dictionaries[index]['QN']

    def linemixing(self, index):
        """Return line mixing entries for line at index
        """
        return self._dictionaries[index]['LM']

    def _error_in_length_message(self):
        return "Mis-matching length of vectors/lists storing line information"

    def as_ArrayOfLineRecord(self):
        """Turns ARTSCAT5 into array of line records that can be stored to
        file
        """
        out = []
        for i in range(self._n):
            out.append(self.get_arts_str(i))
        return ArrayOfLineRecord(data=out, version='ARTSCAT-5')

    def changeForQN(self, kind='change', qid=None,
                    spec=None, afgl=None, qns=None, information=None):
        """Change information of a line according to identifiers

        Input:
            kind (str): kind of operation to be applied, either 'change' for
            overwriting, 'add' for addition (+=), 'sub' for subtraction (-=),
            'remove' to remove matching lines, or 'keep' to only keep matching
            lines

            qid (QuantumIdentifier): Identifier to the transition or energy
            level as defined in ARTS

            spec (str or NoneType):  Name of species for which the operation
            applies.  None means for all species.  Must be None if qid is given

            afgl (int or NoneType):  AFGL isotopologue integer for which the
            operation applies.  None means for all isotopologue.  Must be None
            if qid is given

            qns (dict, None, QuantumNumberRecord, QuantumNumbers):  The quantum
            numbers for which the operation applies.  None means all quantum
            numbers.  Can be level or transition.  Must be None if qid is given

            information (dict or NoneType):  None for kind 'remove'. dict
            otherwise.  Keys in ARTSCAT5.LineRecordData for non-dictionaries.
            Use 'QN' for quantum numbers, 'LM' for line mixing, and 'PB' for
            pressure-broadening.  If level QN-key, the data is applied for both
            levels if they match (only for 'QN'-data)

        Output:
            None, only changes the class instance itself
        """
        if qid is not None:
            assert spec is None and afgl is None and qns is None, \
                "Only one of qid or spec, afgl, and qns combinations allowed"
            spec = qid.species
            afgl = qid.afgl
            qns = as_quantumnumbers(qns)
        else:
            qns = as_quantumnumbers(qns)

        if kind == 'remove':
            assert information is None, "information not None for 'remove'"
            remove_these = []
            remove = True
            change = False
            add = False
            sub = False
            keep = False
        elif kind == 'change':
            assert type(information) is dict, "information is not dictionary"
            remove = False
            change = True
            add = False
            sub = False
            keep = False
        elif kind == 'add':
            assert type(information) is dict, "information is not dictionary"
            remove = False
            change = False
            add = True
            sub = False
            keep = False
        elif kind == 'sub':
            assert type(information) is dict, "information is not dictionary"
            remove = False
            change = False
            add = False
            sub = True
            keep = False
        elif kind == 'keep':
            assert information is None, "information not None for 'keep'"
            remove_these = []
            remove = False
            change = False
            add = False
            sub = False
            keep = True
        assert remove or change or add or keep or sub, "Invalid kind"

        # Check if the quantum number information is for level or transition
        if type(qns) is QuantumNumberRecord:
            for_transitions = True
        else:
            for_transitions = False

        if information is not None:
            for key in information:
                assert key in self.LineRecordData or \
                       key in ['QN', 'LM', 'PB'], \
                       "Unrecognized key"

        # Looping over all the line data
        for i in range(self._n):

            # If spec is None, then all species, otherwise this should match
            if spec is not None:
                if not spec == self.LineRecordData['spec'][i]:
                    continue

            # If afgl is None, then all isotopes, otherwise this should match
            if afgl is not None:
                if not afgl == self.LineRecordData['afgl'][i]:
                    continue

            # Test which levels match and which do not --- partial matching
            test = self.quantumnumbers(i) >= qns
            if for_transitions:
                test = [test]  # To let all and any work

            # Append lines to remove later (so indexing is not messed up)
            if remove and all(test):
                remove_these.append(i)
                continue
            elif keep and not all(test):
                remove_these.append(i)
                continue
            elif keep or remove:
                continue

            # Useless to continue past this point if nothing matches
            if not all(test) and for_transitions:
                continue
            elif not any(test):
                continue

            # There should only be matches remaining but only QN info is level-
            # based so all other infromation must be perfect match
            for info_key in information:
                info = information[info_key]
                if info_key == 'QN':
                    if for_transitions:
                        if change:
                            self._dictionaries[i]['QN'] = info
                        elif add:
                            self._dictionaries[i]['QN'] += info
                        elif sub:
                            self._dictionaries[i]['QN'] -= info
                        else:
                            assert False, "Programmer error?"
                    else:
                        if test[0]:
                            if change:
                                self._dictionaries[i]['QN']['UP'] = info
                            elif add:
                                self._dictionaries[i]['QN']['UP'] += info
                            elif sub:
                                self._dictionaries[i]['QN']['UP'] -= info
                            else:
                                assert False, "Programmer error?"
                        if test[1]:
                            if change:
                                self._dictionaries[i]['QN']['LO'] = info
                            elif add:
                                self._dictionaries[i]['QN']['LO'] += info
                            elif sub:
                                self._dictionaries[i]['QN']['LO'] -= info
                            else:
                                assert False, "Programmer error?"
                elif info_key in ['PB', 'LM']:
                    if not all(test):
                        continue
                    if change:
                        self._dictionaries[i][info_key] = info
                    elif add:
                        assert info['Type'] == \
                            self._dictionaries[i][info_key]['Type'], \
                            "Can only add to matching type"
                        self._dictionaries[i][info_key]['Data'] += info
                    elif sub:
                        assert info['Type'] == \
                            self._dictionaries[i][info_key]['Type'], \
                            "Can only sub from matching type"
                        self._dictionaries[i][info_key]['Data'] -= info
                    else:
                        assert False, "Programmer error?"
                else:
                    if not all(test):
                        continue
                    if change:
                        self.LineRecodData[info_key][i] = info
                    elif add:
                        self.LineRecodData[info_key][i] += info
                    elif sub:
                        self.LineRecodData[info_key][i] -= info
                    else:
                        assert False, "Programmer error?"

        # Again, to not get into index problems, this loop is reversed
        if remove or keep:
            for i in remove_these[::-1]:
                self.remove_line(i)

    def remove_line(self, index):
        """Remove line at index from line record
        """
        assert index < self._n, "index out of bounds"
        assert index > -1, "index must be above 0"

        for key in self.LineRecordData:
            t1 = self.LineRecordData[key][:index]
            t2 = self.LineRecordData[key][(index+1):]
            self.LineRecordData[key] = np.append(t1, t2)

        _t = self._dictionaries
        t1 = _t[:index]
        t2 = _t[(index+1):]
        self._dictionaries = np.append(t1, t2)

        self._n -= 1
        self._assert_sanity_()

    def write_xml(self, xmlwriter, attr=None):
        """Write an ARTSCAT5 object to an ARTS XML file.
        """
        tmp = self.as_ArrayOfLineRecord()
        tmp.write_xml(xmlwriter, attr=attr)


class Rational(_R):
    """Rational number

    This is a copy of fractions.Fraction with only the __repr__ function over-
    written to match ARTS style.  That is 3/2 is represented as such rather
    than as "Fraction(3, 2)".  See original class for more information,
    limitations, and options
    """
    def __init__(self, *args):
        super(Rational, self).__init__()

    def __repr__(self):
        return str(self.numerator) + '/' + str(self.denominator)
    _R.__repr__ = __repr__


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

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        if self.version == 1:
            self._data_dict = {}
            self._keys = {}
            for ii in range(len(data)):
                iso_data = data[ii]
                tmp = iso_data.split()
                self._keys[tmp[1]] = ii
                self._data_dict[tmp[1]] = float(tmp[2])
        elif self.version == 2:
            self._data_dict = {}
            self._keys = {}
            for ii in range(len(data)):
                tmp = data[ii]
                self._keys[tmp[0]] = ii
                self._data_dict[tmp[0]] = [tmp[1], tmp[2]]

    def __getitem__(self, key):
        return self._data_dict[key]

    def __setitem__(self, key, val):
        self._data_dict[key] = val
        if self.version == 1:
            self._data[(self._keys[key])] = '@ ' + key + ' ' + str(val)
        elif self.version == 2:
            self._data[(self._keys[key])] = val

    def species(self):
        return list(self._data_dict.keys())

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a SpeciesAuxData object from an existing file.
        """

        version = int(xmlelement.attrib['version'])

        if version == 1:
            nparam = int(xmlelement.attrib['nparam'])
            data = [s for s in xmlelement.text.split('\n') if s != '']
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
        else:
            raise Exception(
                "Unknown SpeciesAuxData version {}.".format(version))

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

    def __init__(self,
                 speciestags=None,
                 nonlinearspecies=None,
                 frequencygrid=None,
                 pressuregrid=None,
                 referencevmrprofiles=None,
                 referencetemperatureprofile=None,
                 temperaturepertubations=None,
                 nonlinearspeciesvmrpertubations=None,
                 absorptioncrosssection=None):

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


class Sparse(scipy.sparse.csc_matrix):
    """Wrapper around :class:`scipy.sparse.csc_matrix`.

    This class wraps around the SciPy Compressed Sparse Column matrix. The
    usage is exactly the same, but support for reading and writing XML files
    is added. Also additional attributes were added, which follow the ARTS
    names.

    See ARTS_ and SciPy_ documentations for more details.

    .. _ARTS: http://arts.mi.uni-hamburg.de/docserver-trunk/groups/Sparse
    .. _SciPy: http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html

    """
    @property
    def nrows(self):
        """Number of rows."""
        return self.shape[0]

    @property
    def ncols(self):
        """Number of columns."""
        return self.shape[0]

    @property
    def rowindex(self):
        """Row indices to locate data in matrix."""
        return self.tocoo().row

    @property
    def colindex(self):
        """Column indices to locate data in matrix."""
        return self.tocoo().col

    @property
    def sparsedata(self):
        """Data value at specified positions in matrix."""
        return self.tocoo().data

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a Sparse object from an existing file."""

        binaryfp = xmlelement.binaryfp
        nelem = int(xmlelement[0].attrib['nelem'])
        nrows = int(xmlelement.attrib['nrows'])
        ncols = int(xmlelement.attrib['ncols'])

        if binaryfp is None:
            rowindex = np.fromstring(xmlelement[0].text, sep=' ').astype(int)
            colindex = np.fromstring(xmlelement[1].text, sep=' ').astype(int)
            sparsedata = np.fromstring(xmlelement[2].text, sep=' ')
        else:
            rowindex = np.fromfile(binaryfp, dtype='<i4', count=nelem)
            colindex = np.fromfile(binaryfp, dtype='<i4', count=nelem)
            sparsedata = np.fromfile(binaryfp, dtype='<d', count=nelem)

        return cls((sparsedata, (rowindex, colindex)), [nrows, ncols])

    def write_xml(self, xmlwriter, attr=None):
        """Write a Sparse object to an ARTS XML file."""

        # Get ARTS-style information from CSC matrix.
        nrows = self.shape[0]
        ncols = self.shape[1]
        rowindex = self.tocoo().row
        colindex = self.tocoo().col
        sparsedata = self.tocoo().data

        precision = xmlwriter.precision

        if attr is None:
            attr = {}

        attr['nrows'] = nrows
        attr['ncols'] = ncols

        xmlwriter.open_tag('Sparse', attr)

        binaryfp = xmlwriter.binaryfilepointer

        if binaryfp is None:
            xmlwriter.open_tag('RowIndex', {'nelem': rowindex.size})
            for i in rowindex:
                xmlwriter.write('%d' % i + '\n')
            xmlwriter.close_tag()
            xmlwriter.open_tag('ColIndex', {'nelem': colindex.size})
            for i in colindex:
                xmlwriter.write('%d' % i + '\n')
            xmlwriter.close_tag()
            xmlwriter.open_tag('SparseData', {'nelem': sparsedata.size})
            for i in sparsedata:
                xmlwriter.write(('%' + precision) % i + '\n')
            xmlwriter.close_tag()
            xmlwriter.close_tag()
        else:
            xmlwriter.open_tag('RowIndex', {'nelem': rowindex.size})
            np.array(rowindex, dtype='i4').tofile(binaryfp)
            xmlwriter.close_tag()
            xmlwriter.open_tag('ColIndex', {'nelem': colindex.size})
            np.array(colindex, dtype='i4').tofile(binaryfp)
            xmlwriter.close_tag()
            xmlwriter.open_tag('SparseData', {'nelem': sparsedata.size})
            np.array(sparsedata, dtype='d').tofile(binaryfp)
            xmlwriter.close_tag()
            xmlwriter.close_tag()


class QuantumIdentifier:
    """Represents a QuantumIdentifier object.

    See online ARTS documentation for object details.

    """

    def __init__(self, qid):

        assert type(qid) is str, "Need String input"
        these = qid.split()
        assert len(these) > 0, "No QuantumIdentifier"
        spec = these[0].split('-')  # UGLY: What about negative charge?
        if len(spec) == 1:
            self._afgl = None
            if spec[0] == 'None':
                self._spec = None
            else:
                self._spec = spec[0]
        elif len(spec) == 2:
            self._spec = spec[0]
            self._afgl = int(spec[1])
        else:
            assert False, "Cannot recognize species"

        if len(these) == 1:
            self._transition = False
            self._level = False
            return

        if these[1] == 'TR':
            self._transition = True
            self._level = False
        elif these[1] == 'EN':
            self._transition = False
            self._level = True
        else:
            assert False, "Must be energy level [EN] or transition [TR] type"

        self._qns = as_quantumnumbers(" ".join(these[2:]))

        self._assert_sanity_()

    def __repr__(self):
        out = str(self._spec)
        if self._afgl is not None:
            out += '-' + str(self._afgl)
        if self._transition or self._level:
            if self._transition:
                out += ' TR '
            else:
                out += ' EN '
            out += str(self._qns)
        return out

    def _assert_sanity_(self):
        if self._transition:
            assert type(self._qns) is QuantumNumberRecord, "Mismatching types"
        elif self._level:
            assert type(self._qns) is QuantumNumbers, "Mismatching types"
        else:
            assert False, "Programmer error?"

    def __str__(self):
        assert self.afgl is not None or self.species is not None, \
            "Bad data cannot be converted to str.  Contains no species or iso"
        return self.__repr__()

    @property
    def qns(self):
        return self._qns

    @qns.setter
    def qns(self, qns):
        self._qns = as_quantumnumbers(qns)
        if type(self._qns) is QuantumNumberRecord:
            self._transition = True
            self._level = False
        elif type(self._qns) is QuantumNumbers:
            self._transition = False
            self._level = True
        else:
            assert False, "Programmer error?"

    @property
    def species(self):
        return self._spec

    @species.setter
    def species(self, value):
        self._spec = return_if_arts_type(value, 'String')

    @property
    def afgl(self):
        return self._afgl

    @afgl.setter
    def afgl(self, value):
        self._afgl = return_if_arts_type(value, 'Index')

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
        xmlwriter.write(self.__str__())
        xmlwriter.close_tag()


class QuantumNumberRecord:
    """Represents a QuantumNumberRecord object.

    See online ARTS documentation for object details.

    """

    def __init__(self, upper=None, lower=None):
        self._qns = {'UP': QuantumNumbers(), 'LO': QuantumNumbers()}
        self._qns['UP'] = return_if_arts_type(upper, 'QuantumNumbers')
        self._qns['LO'] = return_if_arts_type(lower, 'QuantumNumbers')

    def __repr__(self):
        return "UP " + str(self._qns['UP']) + " LO " + str(self._qns['LO'])

    def __str__(self):
        if self._qns['UP'] is None or self._qns['UP'] is None:
            return ''
        else:
            return self.__repr__()

    def __getitem__(self, key):
        return self._qns[key]

    def __setitem__(self, key, value):
        self._qns[key] = return_if_arts_type(as_quantumnumbers(value),
                                             'QuantumNumbers')

    def __iter__(self):
        return iter(self._qns)

    def __contains__(self, value):
        return value in ['UP', 'LO']

    def from_dict(dict):
        """Creates a QuantumNumberRecord from dict
        """
        if len(dict) == 0:
            return QuantumNumberRecord(upper=QuantumNumbers(),
                                       lower=QuantumNumbers())

        assert 'UP' in dict and 'LO' in dict, "Need UP and LO to create"
        qnr = QuantumNumberRecord(upper=QuantumNumbers(dict['UP']),
                                  lower=QuantumNumbers(dict['LO']))
        return qnr

    def from_str(str):
        """Creates a QuantumNumberRecord from dict
        """
        str = str.strip()
        if len(str) == 0:
            return QuantumNumberRecord(upper=QuantumNumbers(),
                                       lower=QuantumNumbers())

        assert 'UP' in str and 'LO' in str, "Need UP and LO to create"
        _t1 = str.split('UP')
        assert len(_t1) == 2, "Unexpectedly many/few UP in str"
        if len(_t1[0]) == 0:
            _t2 = _t1[1].split('LO')
            assert len(_t2) == 2, "Unexpectedly many/few LO in str"
            lo = _t2[1]
            up = _t2[0]
        else:
            up = _t1[1]
            _t2 = _t1[0].split('LO')
            assert len(_t2) == 2, "Unexpectedly many/few LO in str"
            lo = _t2[1]

        qnr = QuantumNumberRecord(upper=QuantumNumbers(up),
                                  lower=QuantumNumbers(lo))
        return qnr

    @property
    def upper(self):
        """QuantumNumbers object representing the upper quantumnumber."""
        return self._qns['UP']

    @property
    def lower(self):
        """QuantumNumbers object representing the lower quantumnumber."""
        return self._qns['LO']

    @upper.setter
    def upper(self, upper):
        self._qns['UP'] = return_if_arts_type(upper, 'QuantumNumbers')

    @lower.setter
    def lower(self, lower):
        self._qns['LO'] = return_if_arts_type(lower, 'QuantumNumbers')

    @property
    def qns(self):
        return self._qns

    @qns.setter
    def qns(self, value):
        if 'LO' in value:
            self._qns['LO'] = QuantumNumbers(value['LO'])
        else:
            self._qns['LO'] = QuantumNumbers()

        if 'UP' in value:
            self._qns['UP'] = QuantumNumbers(value['UP'])
        else:
            self._qns['UP'] = QuantumNumbers()

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

    def __iadd__(self, qnr):
        self._qns['UP'] += qnr['UP']
        self._qns['LO'] += qnr['LO']

    def __isub__(self, qnr):
        self._qns['UP'] -= qnr['UP']
        self._qns['LO'] -= qnr['LO']

    def __eq__(self, qns):
        if type(qns) is QuantumNumberRecord:
            return self['LO'] == qns['LO'] and self['UP'] == qns['UP']
        elif type(qns) is QuantumNumbers:
            return self['UP'] == qns, self['LO'] == qns
        else:
            return self == as_quantumnumbers(qns)

    def __ne__(self, qns):
        if type(qns) is QuantumNumberRecord:
            return self['LO'] != qns['LO'] and self['UP'] != qns['UP']
        elif type(qns) is QuantumNumbers:
            return self['UP'] != qns, self['LO'] != qns
        else:
            return self != as_quantumnumbers(qns)

    def __lt__(self, qns):
        if type(qns) is QuantumNumberRecord:
            return self['LO'] < qns['LO'] and self['UP'] < qns['UP']
        elif type(qns) is QuantumNumbers:
            return self['UP'] < qns, self['LO'] < qns
        else:
            return self < as_quantumnumbers(qns)

    def __gt__(self, qns):
        if type(qns) is QuantumNumberRecord:
            return self['LO'] > qns['LO'] and self['UP'] > qns['UP']
        elif type(qns) is QuantumNumbers:
            return self['UP'] > qns, self['LO'] > qns
        else:
            return self > as_quantumnumbers(qns)

    def __le__(self, qns):
        if type(qns) is QuantumNumberRecord:
            return self['LO'] <= qns['LO'] and self['UP'] <= qns['UP']
        elif type(qns) is QuantumNumbers:
            return self['UP'] <= qns, self['LO'] <= qns
        else:
            return self <= as_quantumnumbers(qns)

    def __ge__(self, qns):
        if type(qns) is QuantumNumberRecord:
            return self['LO'] >= qns['LO'] and self['UP'] >= qns['UP']
        elif type(qns) is QuantumNumbers:
            return self['UP'] >= qns, self['LO'] >= qns
        else:
            return self >= as_quantumnumbers(qns)


def as_quantumnumbers(var):
    """Takes a quantum number prospect and turns it into a quantum number type
    if possible

    Parameters:
        var (dict, QuantumNumberRecord, QuantumNumbers, None, str): Quantum
        numbers

    Returns:
        QN (QuantumNumberRecord, QuantumNumbers): Returned quantumn numbers.
        No change if already quantum numbers type
    """

    if type(var) in [QuantumNumberRecord, QuantumNumbers]:
        return var
    elif var is None:
        return QuantumNumbers()

    assert type(var) in [dict, str], "Cannot recognize as quantum number"

    if 'UP' in var and 'LO' in var:
        if type(var) is dict:
            return QuantumNumberRecord.from_dict(var)
        else:
            return QuantumNumberRecord.from_str(var)
    elif 'UP' in var:
        if type(var) is dict:
            var['LO'] = {}
            return QuantumNumberRecord.from_dict(var)
        else:
            return QuantumNumberRecord.from_str(var + ' LO')
    elif 'LO' in var:
        if type(var) is dict:
            var['UP'] = {}
            return QuantumNumberRecord.from_dict(var)
        else:
            return QuantumNumberRecord.from_str(var + ' UP')
    else:
        return QuantumNumbers(var)


class QuantumNumbers:
    """Represents a QuantumNumbers object.

    See online ARTS documentation for object details.

    """

    def __init__(self, numbers=None, nelem=None):

        self.numbers = numbers
        if nelem is not None:
            self.nelem = nelem
        else:
            self.nelem = len(self.numbers)

        self._assert_sanity_()

    def _assert_sanity_(self):
        if self.nelem is None or self.numbers is None:
            return
        assert len(self.numbers) == self.nelem, "mismatching quantum numbers"

    def __repr__(self):
        out = ''
        for qn in self.numbers:
            out += qn + ' ' + str(self.numbers[qn]) + ' '
        return out[:-1]

    def __getitem__(self, key):
        """Returns the value.  Mimics ARTS' behavior for mismatched data
        """
        if key in self:
            return self.numbers[key]
        else:
            return None

    def __setitem__(self, key, value):
        """Sets a value and counts up the quantum numbers
        """
        if key in self.numbers:
            self.numbers[key] = Rational(value)
        else:
            self.numbers[key] = Rational(value)
            self.nelem += 1
        self._assert_sanity_()

    def __iadd__(self, qns):
        for qn in qns:
            assert qn not in self, "Addition means adding new QN. Access " + \
                "individual elements to change their values"
            self.numbers[qn] = qns[qn]
            self.nelem += 1
        return self

    def __isub__(self, qns):
        for qn in qns:
            assert qn in self, "Subtraction means removing QN. Access " + \
                "individual elements to change their values"
            del self.numbers[qn]
            self.nelem -= 1
        return self

    def __contains__(self, key):
        """Are these quantum numbers here?
        """
        return key in self.numbers

    def __iter__(self):
        return iter(self.numbers)

    def __eq__(self, qns):
        """Tests for complete equality
        """
        return self <= qns and len(qns) == self.nelem

    def __ne__(self, qns):
        return not self.__eq__(qns)

    def __le__(self, qns):
        """Tests for all in self being in qns
        """
        try:
            for qn in self:
                if qns[qn] != self[qn]:
                    return False
            return True
        except:
            return False

    def __ge__(self, qns):
        """Tests for all in qns being in self
        """
        try:
            for qn in qns:
                if qns[qn] != self[qn]:
                    return False
            return True
        except:
            return False

    def __lt__(self, qns):
        """Tests for all in self being in qns and if there is more in qns
        """
        return self <= qns and self.nelem < len(qns)

    def __gt__(self, qns):
        """Tests for all in self being in qns and if there is more in self
        """
        return qns <= self and len(qns) < self.nelem

    def __len__(self):
        return self.nelem

    def array_of_M(self):
        """Returns all possible M in a list.  Requires presence of J
        """
        assert 'J' in self, "Need J to define M"
        assert self['J'] >= 0, "Negative J in this instance?"
        _t = []
        _s = -self['J']
        while _s <= self['J']:
            _t.append(_s)
            _s += 1
        return np.array(_t)

    @property
    def numbers(self):
        """Dict representing the quantumnumbers."""
        return self._numbers

    @property
    def nelem(self):
        """Number of quantumnumbers stored."""
        return self._nelem

    @numbers.setter
    def numbers(self, numbers):
        if type(numbers) is str:
            _t = numbers.split()
            nums = {}
            i = 0
            assert len(_t) % 2 == 0, "Not of form 'key1 value1 key2 value2'"
            while i < len(_t):
                nums[_t[i]] = Rational(_t[i+1])
                i += 2
            self._numbers = nums
        elif type(numbers) is dict:
            for i in numbers:
                numbers[i] = Rational(numbers[i])
            self._numbers = numbers
        elif type(numbers) is QuantumNumbers:
            self._numbers = numbers.numbers
        elif numbers is None:
            self._numbers = {}
        else:
            print(numbers)
            assert False, "Expected dict or String for QuantumNumbers"
        # OLD: self._numbers = return_if_arts_type(numbers, 'String')

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
        xmlwriter.write(self.__str__())
        xmlwriter.close_tag(newline=False)


class LineMixingRecord:
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
        self._data = return_if_arts_type(data, 'Vector')

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
