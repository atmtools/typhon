# -*- coding: utf-8 -*-
"""
Implementation of classes to handle various catalogue information.

"""

try:
    from .utils import return_if_arts_type
    from .. import constants
    from .. import spectroscopy
except:
    from typhon.arts.utils import return_if_arts_type
    import typhon.constants as constants
    import typhon.spectroscopy as spectroscopy


import numpy as np
import scipy.sparse
import scipy.interpolate as _ip
from scipy.special import wofz as _Faddeeva_
from fractions import Fraction as _R
from numpy.polynomial import Polynomial as _P

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
        return ARTSCAT5(self)

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

    def __init__(self, init_data=None):
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

        if init_data is None:
            return

        self.append(init_data, sort=False)

    def _append_linestr_(self, linerecord_str):
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
        self._dictionaries[-1]['LM'] = LineMixing(self._dictionaries[-1]['LM'])
        self._dictionaries[-1]['PB'] = \
            PressureBroadening(self._dictionaries[-1]['PB'])

    def _append_line_(self, line):
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

    def _append_ArrayOfLineRecord_(self, array_of_linerecord, sort=True):
        """Appends lines in ArrayOfLineRecord to ARTSCAT5

        By default sorts all lines by frequency before finishing
        """
        assert array_of_linerecord.version == 'ARTSCAT-5', "Only for ARTSCAT-5"
        for l in array_of_linerecord:
            self._append_linestr_(l)

    def _append_ARTSCAT5_(self, artscat5, sort=True):
        """Appends all the lines of another artscat5 to this

        By default sorts all lines by frequency before finishing
        """
        for line in artscat5:
            self._append_line_(line)

    def append(self, other, sort=True):
        """Appends data to ARTSCAT5.  Used at initialization

        Parameters:
            other (str, ARTSCAT5, ArrayOfLineRecord, tuple): Data to append,
            Must fit with internal structures.  Easiest to guarantee if other
            is another ARTSCAT5 or an ArrayOfLineRecord containing ARTSCAT-5
            data

            sort: Sorts the lines by frequency if True
        """
        if type(other) is str:
            self._append_linestr_(other)
        elif type(other) is ARTSCAT5:
            self._append_ARTSCAT5_(other)
        elif type(other) is tuple:  # For lines --- this easily fails
            self._append_line_(other)
        elif type(other) is ArrayOfLineRecord:
            self._append_ArrayOfLineRecord_(other)
        elif type(other) in [list, np.ndarray]:
            for x in other:
                self.append(x)
        else:
            assert False, "Unknown type"
        self._assert_sanity_()
        if sort:
            self.sort()

    def sort(self, kind='freq', ascending=True):
        """Sorts the ARTSCAT5 data by kind.  Set ascending to False for
        descending sorting

        Parameters:
            kind (str): The key to LineRecordData

            ascending (bool): True sorts ascending, False sorts descending

        Examples:
            Sort by descending frequnecy

            >>> cat = typhon.arts.xml.load('C2H2.xml').as_ARTSCAT5()
            >>> cat.LineRecordData['freq']
            array([  5.94503434e+10,   1.18899907e+11,   1.78347792e+11, ...,
                     2.25166734e+12,   2.31051492e+12,   2.36933091e+12])
            >>> cat.sort(ascending=False)
            >>> cat.LineRecordData['freq']
            array([  2.36933091e+12,   2.31051492e+12,   2.25166734e+12, ...,
                     1.78347792e+11,   1.18899907e+11,   5.94503434e+10])

            Sort by line strength

            >>> cat = typhon.arts.xml.load('C2H2.xml').as_ARTSCAT5()
            >>> cat.LineRecordData['str']
            array([  9.02281290e-21,   7.11410308e-20,   2.34380510e-19, ...,
                     4.77325112e-19,   3.56443438e-19,   2.63222798e-19])
            >>> cat.sort(kind='str')
            >>> cat.LineRecordData['str']
            array([  9.02281290e-21,   7.11410308e-20,   2.34380510e-19, ...,
                     1.09266008e-17,   1.10644138e-17,   1.10939452e-17])

        """
        assert kind in self.LineRecordData, "kind must be in LineRecordData"

        i = np.argsort(self.LineRecordData[kind])
        if not ascending:
            i = i[::-1]

        for key in self.LineRecordData:
            self.LineRecordData[key] = self.LineRecordData[key][i]
        self._dictionaries = self._dictionaries[i]

    def remove(self, upper_limit=None, lower_limit=None, kind='freq'):
        """Removes lines not within limits of kind

        This loops over all lines in self and only keeps those fulfilling

        .. math::
            l \\leq x \\leq u,

        where l is a lower limit, u is an upper limit, and x is a parameter
        in self.LineRecordData

        Parameters:
            upper_limit (float): value to use for upper limit [-]

            lower_limit (float): value to use for lower limit [-]

            kind (str): keyword for determining x.  Must be key in
            self.LineRecordData

        Returns:
            None: Only changes self

        Examples:
            Remove lines below 1 THz and above 1.5 THz

            >>> cat = typhon.arts.xml.load('C2H2.xml').as_ARTSCAT5()
            >>> cat
            ARTSCAT-5 with 40 lines. Species: ['C2H2']
            >>> cat.remove(lower_limit=1000e9, upper_limit=1500e9)
            >>> cat
            ARTSCAT-5 with 9 lines. Species: ['C2H2']

            Remove weak lines

            >>> cat = typhon.arts.xml.load('C2H2.xml').as_ARTSCAT5()
            >>> cat
            ARTSCAT-5 with 40 lines. Species: ['C2H2']
            >>> cat.remove(lower_limit=1e-18, kind='str')
            >>> cat
            ARTSCAT-5 with 31 lines. Species: ['C2H2']
        """
        assert upper_limit is not None or lower_limit is not None, \
            "Cannot remove lines when the limits are undeclared"
        assert kind in self.LineRecordData, "Needs kind in LineRecordData"
        remove_these = []
        for i in range(self._n):
            if lower_limit is not None:
                if self.LineRecordData[kind][i] < lower_limit:
                    remove_these.append(i)
                    continue
            if upper_limit is not None:
                if self.LineRecordData[kind][i] > upper_limit:
                    remove_these.append(i)
                    continue

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
        text = str(self.pressurebroadening(index))
        if len(text) > 0:
            s += ' PB ' + text
        text = str(self.quantumnumbers(index))
        if len(text) > 0:
            s += ' QN ' + text
        text = str(self.linemixing(index))
        if len(text) > 0:
            s += ' LM ' + text
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
        """Turns ARTSCAT5 into ArrayOfLineRecord that can be stored to file
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

        Examples:
            Add S = 1 to both levels quantum numbers by adding information to
            all lines

            >>> cat = typhon.arts.xml.load('O2.xml').as_ARTSCAT5()
            >>> cat.quantumnumbers(0)
            UP v1 0 J 32 F 61/2 N 32 LO v1 0 J 32 F 59/2 N 32
            >>> cat.changeForQN(information={'QN': {'S': 1}}, kind='add')
            >>> cat.quantumnumbers(0)
            UP S 1 v1 0 J 32 F 61/2 N 32 LO S 1 v1 0 J 32 F 59/2 N 32

            Remove all lines not belonging to a specific isotopologue and band
            by giving the band quantum numbers

            >>> cat = typhon.arts.xml.load('O2.xml').as_ARTSCAT5()
            >>> cat
            ARTSCAT-5 with 6079 lines. Species: ['O2']
            >>> cat.changeForQN(kind='keep', afgl=66, qns={'LO': {'v1': 0},
                                                           'UP': {'v1': 0}})
            >>> cat
            ARTSCAT-5 with 187 lines. Species: ['O2']

            Change the frequency of the 119 GHz line to 3000 THz by giving a
            full and unique quantum number match

            >>> cat = typhon.arts.xml.load('O2.xml').as_ARTSCAT5()
            >>> cat.sort()
            >>> cat.LineRecordData['freq']
            array([  9.00e+03,   2.35e+04,   4.01e+04, ...,   2.99e+12,
                     2.99e+12,   2.99e+12])
            >>> cat.changeForQN(afgl=66, qns={'LO': {'v1': 0, 'J': 0, 'N': 1},
                                              'UP': {'v1': 0, 'J': 1, 'N': 1}},
                                information={'freq': 3000e9})
            >>> cat.sort()
            >>> cat.LineRecordData['freq']
            array([  9.00e+03,   2.35e+04,   4.01e+04, ...,   2.99e+12,
                     2.99e+12,   3.00e+12])
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
                    if keep:
                        remove_these.append(i)
                    continue

            # If afgl is None, then all isotopes, otherwise this should match
            if afgl is not None:
                if not afgl == self.LineRecordData['afgl'][i]:
                    if keep:
                        remove_these.append(i)
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
                        assert info.kind == \
                            self._dictionaries[i][info_key].kind, \
                            "Can only add to matching type"
                        self._dictionaries[i][info_key].data += info
                    elif sub:
                        assert info.kind == \
                            self._dictionaries[i][info_key].kind, \
                            "Can only sub from matching type"
                        self._dictionaries[i][info_key].data -= info
                    else:
                        assert False, "Programmer error?"
                else:
                    if not all(test):
                        continue
                    if change:
                        self.LineRecordData[info_key][i] = info
                    elif add:
                        self.LineRecordData[info_key][i] += info
                    elif sub:
                        self.LineRecordData[info_key][i] -= info
                    else:
                        assert False, "Programmer error?"

        # Again, to not get into index problems, this loop is reversed
        if remove or keep:
            for i in remove_these[::-1]:
                self.remove_line(i)

    def remove_line(self, index):
        """Remove line at index from line record
        """
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

    def cross_section(self, temperature=None, pressure=None,
                      vmrs=None, mass=None, isotopologue_ratios=None,
                      partition_functions=None, f=None):
        """Provides an estimation of the cross-section in the provided
        frequency range

        Computes the following estimate (summing over all lines):

        .. math::
            \\sigma(f) = \\sum_{k=0}^{k=n-1}
            r_k S_{0, k}(T_0) K_1 K_2 \\frac{Q(T_0)}{Q(T)}
            \\frac{1 + G_k \\; p^2 + iY_k \\; p }{\\gamma_{D,k}\\sqrt{\\pi}}
            \\; F\\left(\\frac{f - f_{0,k} -  \Delta f_k \\; p^2 -
            \\delta f_kp + i\\gamma_{p,k}p} {\\gamma_{D,k}}\\right),

        where there are n lines,
        r is the isotopologue ratio,
        S_0 is the line strength,
        K_1 is the boltzman level statistics,
        K_2 is the stimulated emission,
        Q is the partition sum, G is the second
        order line mixing coefficient,
        p is pressure,
        Y is the first order line mixing coefficient,
        f_0 is the line frequency,
        Delta-f is the second order line mixing frequency shift,
        delta-f is the first order pressure shift,
        gamma_p is the pressure broadening half width,
        gamma_D is the Doppler half width, and
        F is assumed to be the Faddeeva function

        Note 1: this is only meant for quick-and-dirty estimates.  If data is
        lacking, very simplistic assumptions are made to complete the
        calculations.
        Lacking VMR for a species assumes 1.0 vmr of the line species itself,
        lacking mass assumes dry air mass,
        lacking isotopologue ratios means assuming a ratio of unity,
        lacking partition functions means the calculations are performed at
        line temperatures,
        lacking frequency means computing 1000 frequencies from lowest
        frequency line minus its pressure broadening to the highest frequency
        line plus its pressure broadening,
        lacking pressure means computing at 1 ATM, and
        lacking temperature assumes atmospheric temperatures the same as the
        first line temperature.  If input f is None then the return
        is (f, sigma), else the return is (sigma)

        Warning: Use only as an estimation, this function is neither optimized
        and is only tested for a single species in arts-xml-data to be within
        1% of the ARTS computed value

        Parameters:
            temperature (float): Temperature [Kelvin]

            pressure (float): Pressure [Pascal]

            vmrs (dict-like): Volume mixing ratios.  See PressureBroadening for
            use [-]

            mass (dict-like): Mass of isotopologue [kg]

            isotopologue_ratios (dict-like):  Isotopologue ratios of the
            different species [-]

            partition_functions (dict-like):  Partition function estimator,
            should compute partition function by taking temperature as the only
            argument [-]

            f (ndarray): Frequency [Hz]

        Returns:
            (f, xsec) or xsec depending on f

        Examples:
            Plot cross-section making no assumptions on the atmosphere or
            species, i.e., isotopologue ratios is 1 for all isotopologue
            (will not agree with ARTS)

            >>> import matplotlib.pyplot as plt
            >>> cat = typhon.arts.xml.load('O2.xml').as_ARTSCAT5()
            >>> (f, x) = cat.cross_section()
            >>> plt.plot(f, x)

            Plot cross-sections by specifying limited information on the
            species (will agree reasonably with ARTS)

            >>> import matplotlib.pyplot as plt
            >>> cat = typhon.arts.xml.load('O2.xml').as_ARTSCAT5()
            >>> cat.changeForQN(afgl=66, kind='keep')
            >>> f, x = cat.cross_section(mass={"O2-66": 31.9898*constants.amu},
                                         isotopologue_ratios={"O2-66": 0.9953})
            >>> plt.plot(f, x)

        """
        if temperature is None:
            temperature = self.LineRecordData['t0'][0]

        if pressure is None:
            pressure = constants.atm

        if vmrs is None:
            vmrs = {}

        if mass is None:
            mass = {}

        if f is None:
            return_f = True
            f0 = self.pressurebroadening(0).compute_pressurebroadening_params(
                    temperature, self.LineRecordData['t0'][0],
                    pressure, vmrs)[0]
            f0 = self.LineRecordData['freq'][0] - f0
            f1 = self.pressurebroadening(-1).compute_pressurebroadening_params(
                    temperature, self.LineRecordData['t0'][-1],
                    pressure, vmrs)[0]
            f1 = self.LineRecordData['freq'][-1] - f1
            f = np.linspace(f0, f1, num=1000)
        else:
            return_f = False

        if isotopologue_ratios is None:
            isotopologue_ratios = {}

        if partition_functions is None:
            partition_functions = {}

        # Cross-section
        sigma = np.zeros_like(f)

        for i in range(self._n):
            spec_key = self.LineRecordData['spec'][i] + '-' + \
                          str(self.LineRecordData['afgl'][i])

            if spec_key in mass:
                m = mass[spec_key]
            else:
                m = constants.molar_mass_dry_air / constants.avogadro
            gamma_D = \
                spectroscopy.doppler_broadening(temperature,
                                                self.LineRecordData['freq'][i],
                                                m)
            (G, Df,
             Y) = self.linemixing(i).compute_linemixing_params(temperature)

            (gamma_p,
             delta_f) = \
                self.pressurebroadening(i).compute_pressurebroadening_params(
                temperature, self.LineRecordData['t0'][i], pressure, vmrs)

            K1 = spectroscopy.boltzmann_level(self.LineRecordData['elow'][i],
                                              temperature,
                                              self.LineRecordData['t0'][i])
            K2 = spectroscopy.stimulated_emission(
                    self.LineRecordData['freq'][i],
                    temperature,
                    self.LineRecordData['t0'][i])

            if spec_key in partition_functions:
                Q = partition_functions[spec_key]
            else:
                Q = np.ones_like

            if spec_key in isotopologue_ratios:
                r = isotopologue_ratios[spec_key]
            else:
                r = 1.0

            S = r * self.LineRecordData['str'][i] * K1 * K2 * \
                Q(self.LineRecordData['t0'][i]) / Q(temperature)

            lm = 1 + G * pressure**2 + 1j * Y * pressure
            z = (f - self.LineRecordData['freq'][i] -
                 delta_f - Df * pressure**2 + 1j * gamma_p) / gamma_D
            sigma += (S * (lm * _Faddeeva_(z) / np.sqrt(np.pi) / gamma_D)).real
        if return_f:
            return f, sigma
        else:
            return sigma

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

    def __repr__(self):
        return "SpeciesAuxData Version " + str(self.version) + ' ' + \
            'for ' + str(len(self.species())) + ' species'

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

    def as_partition_functions(self):
        return partition_functions(self)


class partition_functions:
    """Class to compute partition functions given ARTS-like partition functions
    """
    _default_test = 296.0

    def __init__(self, init_data=None):
        self._data = {}
        self.append(init_data)
        self._assert_sanity_()

    def append(self, data):
        if type(data) is SpeciesAuxData:
            self._from_species_aux_data_(data)
        elif type(data) is dict:
            self._from_dict_(data)
        elif type(data) is list:
            self._from_list_(data)
        elif type(data) is partition_functions:
            self.data = partition_functions.data
        elif data is not None:
            assert False, "Cannot recognize the initialization data type"

    def _from_species_aux_data_(self, sad):
        assert sad.version == 2, "Must be version 2 data"
        self._from_dict_(sad._data_dict)

    def _from_dict_(self, d):
        for k in d:
            assert type(d[k]) is list, "lowest level data must be list"
            self._from_list_(d[k], k)

    def _from_list_(self, l, k):
        if l[0] == 'PART_TFIELD':
            self.data[k] = _ip.interp1d(l[1][0].grids[0], l[1][0].data)
        elif l[0] == 'PART_COEFF':
            self.data[k] = _P(l[1][0].data)
        else:
            raise RuntimeError("Unknown or not implemented " +
                               "partition_functions type encountered")

    def _assert_sanity_(self):
        assert type(self.data) is dict, "Sanity check fail, calss is wrong"

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, data):
        if type(data) is list:
            self._from_list_(data, key)
        elif type(data) in [_P, _ip.interp1d]:
            self.data[key] = data
        else:
            try:
                data(self._default_test)
                self.data[key] = data
            except:
                raise RuntimeError("Cannot determine type")

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "partition functions for " + str(len(self)) + " species"

    def keys(self):
        return self.data.keys()
    species = keys

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        assert type(val) is dict, "new values must be dictionary type"
        self._data = val


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
        if len(self._qns['UP']) == 0 and len(self._qns['LO']) == 0:
            return 'No Quantum-Numbers'
        else:
            return "UP " + str(self._qns['UP']) + " LO " + str(self._qns['LO'])

    def __str__(self):
        if len(self._qns['UP']) == 0 and len(self._qns['LO']) == 0:
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
        """Tests for complete equality ==
        """
        return self <= qns and len(qns) == self.nelem

    def __ne__(self, qns):
        """Tests for lacking complete equality !=
        """
        return not self == qns

    def __le__(self, qns):
        """Tests for all in self being in qns <=
        """
        try:
            for qn in self:
                if qns[qn] != self[qn]:
                    return False
            return True
        except:
            return False

    def __ge__(self, qns):
        """Tests for all in qns being in self >=
        """
        try:
            for qn in qns:
                if qns[qn] != self[qn]:
                    return False
            return True
        except:
            return False

    def __lt__(self, qns):
        """Tests for all in self being in qns and if there is more in qns <
        """
        return self <= qns and self.nelem < len(qns)

    def __gt__(self, qns):
        """Tests for all in self being in qns and if there is more in self >
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
        self.data = LineMixing(data)

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

    def __repr__(self):
        return self.tag + ' ' + str(self.quantumnumberrecord) + ' ' + \
            str(self.data)

    @quantumnumberrecord.setter
    def quantumnumberrecord(self, quantumnumberrecord):
        self._quantumnumberrecord = return_if_arts_type(
            quantumnumberrecord, 'QuantumNumberRecord')

    @data.setter
    def data(self, data):
        self._data = return_if_arts_type(data, 'LineMixing')

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a LineMixingRecord object from an existing file.
        """

        obj = cls()
        obj.tag = xmlelement[0].value()
        obj.quantumnumberrecord = xmlelement[1].value()
        obj.data = LineMixing(xmlelement[2].value())

        return obj

    def write_xml(self, xmlwriter, attr=None):
        """Write a LineMixingRecord object to an ARTS XML file.
        """
        if attr is None:
            attr = {}

        xmlwriter.open_tag("LineMixingRecord", attr)
        xmlwriter.write_xml(self.tag)
        xmlwriter.write_xml(self.quantumnumberrecord)
        xmlwriter.write_xml(self.data.data)
        xmlwriter.close_tag()


class LineMixing:
    """Helper class to hold linemixing data
    """

    _none = None
    _first_order = "L1"
    _second_order = "L2"
    _lblrtm = "LL"
    _lblrtm_nonresonant = "NR"
    _for_band = "BB"
    _possible_kinds = [_none, _first_order, _second_order, _lblrtm,
                       _lblrtm_nonresonant, _for_band]

    def __init__(self, data=None, kind=None):
        self.data = data
        if kind is not None:
            self.kind = kind

        self._assert_sanity_()
        self._make_data_as_in_arts_()

    def _assert_sanity_(self):
        if self._type is self._none:
            assert len(self._data) == 0, "Data available for none-type"
        elif self._type is self._first_order:
            assert len(self._data) == 3, "Data mismatching first order"
        elif self._type is self._second_order:
            assert len(self._data) == 10, "Data mismatching second order"
        elif self._type is self._lblrtm:
            assert len(self._data) == 12, "Data mismatching LBLRTM data"
        elif self._type is self._lblrtm_nonresonant:
            assert len(self._data) == 1, "Data mismatching LBLRTM data"
        elif self._type is self._for_band:
            assert len(self._data) == 1, "Data mismatching band data"
        else:
            assert False, "Cannot recognize data type at all"

    def _make_data_as_in_arts_(self):
        if self._type in [self._none, self._for_band]:
            return
        elif self._type is self._first_order:
            self._t0 = self._data[0]
            self._y0 = self._data[1]
            self._n0 = self._data[2]
        elif self._type is self._second_order:
            self._t0 = self._data[6]

            self._y0 = self._data[0]
            self._y1 = self._data[1]
            self._ey = self._data[7]

            self._g0 = self._data[2]
            self._g1 = self._data[3]
            self._eg = self._data[8]

            self._f0 = self._data[4]
            self._f1 = self._data[5]
            self._ef = self._data[9]
        elif self._type is self._lblrtm:
            self._y = _ip.interp1d(self._data[:4], self._data[4:8])
            self._g = _ip.interp1d(self._data[:4], self._data[8:])
        else:
            assert False, "Unknown data type"

    def __repr__(self):
        out = ''
        if self._type is self._none:
            return "No Line-Mixing"
        elif self._type in self._possible_kinds:
            out += self._type
        else:
            assert False, "Cannot recognize kind"
        for i in self.data:
            out += ' ' + str(i)
        return out

    def __str__(self):
        out = ''
        if self._type is self._none:
            return out
        elif self._type in self._possible_kinds:
            out += self._type
        else:
            assert False, "Cannot recognize kind"
        for i in self.data:
            out += ' ' + str(i)
        return out

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, val):
        self.data[index] = val
        self._make_data_as_in_arts_()

    @property
    def data(self):
        return self._data

    @property
    def kind(self):
        return self._type

    @kind.setter
    def kind(self, val):
        found = False
        for i in self._possible_kinds:
            if i == val:
                self._type = i
                found = True
                break
        assert found, "Cannot recognize kind"

    @data.setter
    def data(self, val):
        self._data = val
        if self._data is None:
            self._data = np.array([], dtype=float)
            self._type = self._none
        elif type(self._data) is dict:
            if self._data['Type'] is None:
                self._type = self._none
            else:
                self.kind = self._data['Type']
            self._data = self._data['Data']
        else:
            if len(self._data) == 10:
                self._type = self._second_order
            elif len(self._data) == 3:
                self._type = self._first_order
            elif len(self._data) == 12:
                self._type = self._lblrtm
            elif len(self._data) == 0:
                self._type = self._none
            else:
                assert False, "Cannot recognize data type automatically"

    def compute_linemixing_params(self, temperature):
        """Returns the line mixing parameters for given temperature(s)

        Cross-section is found from summing all lines

        .. math::
            \\sigma(f) \\propto \sum_{k=0}^{k=n-1}
            \\left[1 + G_k \\; p^2 + iY_k \\; p\\right] \\;
            F\\left(\\frac{f - f_{0,k} -  \Delta f_k \\; p^2 -
            \\delta f_kp + i\\gamma_{p,k}p} {\\gamma_{D,k}}\\right),

        where k indicates line dependent variables.  This function returns
        the line mixing parameters G, Y, and Delta-f.  The non-line
        mixing parameters are gamma_D as the Doppler broadening,  gamma_p
        as the pressure broadening, f as frequency,
        f_0 as the line frequency, delta-f as the first order pressure induced
        frequency shift, and p as pressure.  The function F(···) is the
        Faddeeva function and gives the line shape.  Many scaling factors are
        ignored in the equation above...

        Note 1: that for no line mixing, this function returns all zeroes

        Developer note: the internal variables used emulates the theory for
        each type of allowed line mixing.  Thus it should be easy to extend
        this for other types and for partial derivatives

        Input:
            temperature (float or ndarray) in Kelvin

        Output:
            G(temperature), Delta-f(temperature), Y(temperature)
        """
        if self._type is self._none:
            return np.zeros_like(temperature), np.zeros_like(temperature), \
                np.zeros_like(temperature)
        elif self._type is self._for_band:
            return np.zeros_like(temperature) * np.nan, \
                np.zeros_like(temperature) * np.nan, \
                np.zeros_like(temperature) * np.nan
        elif self._type is self._lblrtm:
            return self._g(temperature), np.zeros_like(temperature), \
                self._y(temperature)
        elif self._type is self._first_order:
            return np.zeros_like(temperature), np.zeros_like(temperature), \
                self._y0 * (self._t0/temperature) ** self._ey
        elif self._type is self._lblrtm_nonresonant:
            return np.full_like(temperature, self._data[0]), \
                np.zeros_like(temperature), np.zeros_like(temperature)
        elif self._type is self._second_order:
            th = self._t0 / temperature
            return (self._g0 + self._g1 * (th - 1)) * th ** self._eg, \
                (self._f0 + self._f1 * (th - 1)) * th ** self._ef, \
                (self._y0 + self._y1 * (th - 1)) * th ** self._ey


class PressureBroadening:
    """Helper class to hold pressurebroadening data
    """

    _none = None
    _air = "N2"
    _air_and_water = "WA"
    _all_planets = "AP"
    _possible_kinds = [_none, _air, _air_and_water, _all_planets]

    def __init__(self, data=None, kind=None):
        self.data = data
        if kind is not None:
            self.kind = kind

        self._assert_sanity_()
        self._make_data_as_in_arts_()

    def _assert_sanity_(self):
        if self._type is self._none:
            assert len(self._data) == 0, "Data available for none-type"
        elif self._type is self._air:
            assert len(self._data) == 10, "mismatching air broadening "
        elif self._type is self._air_and_water:
            assert len(self._data) == 9, "mismatching air and water broadening"
        elif self._type is self._all_planets:
            assert len(self._data) == 20, "mismatching all planets data"
        else:
            assert False, "Cannot recognize data type at all"

    def _make_data_as_in_arts_(self):
        if self._type is self._none:
            return
        elif self._type is self._air:
            self._sgam = self._data[0]
            self._sn = self._data[1]
            self._sdel = 0

            self._agam = self._data[2]
            self._an = self._data[3]
            self._adel = self._data[4]

            self._dsgam = self._data[5]
            self._dnself = self._data[6]

            self._dagam = self._data[7]
            self._dnair = self._data[8]

            self._dadel = self._data[9]
        elif self._type is self._air_and_water:
            self._sgam = self._data[0]
            self._sn = self._data[1]
            self._sdel = self._data[2]

            self._agam = self._data[3]
            self._an = self._data[4]
            self._adel = self._data[5]

            self._wgam = self._data[6]
            self._wn = self._data[7]
            self._wdel = self._data[8]
        elif self._type is self._all_planets:
            self._sgam = self._data[0]
            self._sn = self._data[7]
            self._sdel = 0
            self._gam = {'N2': self._data[1], 'O2': self._data[2],
                         'H2O': self._data[3], 'CO2': self._data[4],
                         'H2': self._data[5], 'He': self._data[6]}
            self._n = {'N2': self._data[8], 'O2': self._data[9],
                       'H2O': self._data[10], 'CO2': self._data[11],
                       'H2': self._data[12], 'He': self._data[13]}
            self._delta_f = {'N2': self._data[14], 'O2': self._data[15],
                             'H2O': self._data[16], 'CO2': self._data[17],
                             'H2': self._data[18], 'He': self._data[19]}
        else:
            assert False, "Unknown data type"

    def __repr__(self):
        out = ''
        if self._type is self._none:
            return "No Pressure-Broadening"
        elif self._type in self._possible_kinds:
            out += self._type
        else:
            assert False, "Cannot recognize kind"
        for i in self.data:
            out += ' ' + str(i)
        return out

    def __str__(self):
        out = ''
        if self._type is self._none:
            return out
        elif self._type in self._possible_kinds:
            out += self._type
        else:
            assert False, "Cannot recognize kind"
        for i in self.data:
            out += ' ' + str(i)
        return out

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, val):
        self.data[index] = val
        self._make_data_as_in_arts_()

    @property
    def data(self):
        return self._data

    @property
    def kind(self):
        return self._type

    @kind.setter
    def kind(self, val):
        found = False
        for i in self._possible_kinds:
            if i == val:
                self._type = i
                found = True
                break
        assert found, "Cannot recognize kind"

    @data.setter
    def data(self, val):
        self._data = val
        if self._data is None:
            self._data = np.array([], dtype=float)
            self._type = self._none
        elif type(self._data) is dict:
            if self._data['Type'] is None:
                self._type = self._none
            else:
                self.kind = self._data['Type']
            self._data = self._data['Data']
        else:
            if len(self._data) == 10:
                self._type = self._air
            elif len(self._data) == 9:
                self._type = self._air_and_water
            elif len(self._data) == 20:
                self._type = self._all_planets
            elif len(self._data) == 0:
                self._type = self._none
            else:
                assert False, "Cannot recognize data type automatically"

    def compute_pressurebroadening_params(self, temperature, line_temperature,
                                          pressure, vmrs):
        """Computes the pressure broadening parameters for the given atmosphere

        Cross-section is found from summing all lines

        .. math::
            \\sigma(f) \\propto \\sum_{k=1}^{k=n-1}
            F\\left(\\frac{f - f_{0,k} -  \Delta f_k \\; p^2 -
            \\delta f_kp + i\\gamma_{p,k}p} {\\gamma_{D,k}}\\right),

        where k indicates line dependent variables.  This function returns
        the pressure broadening parameters p*gamma_p and p*delta-f.  The non-
        pressure broadening parameters are gamma_D as the Doppler broadening,
        f as frequency, f_0 as the line frequency, delta-f as the first order
        pressure induced frequency shift, and p as pressure.  The function
        F(···) is the Faddeeva function and gives the line shape.  Many scaling
        factors are ignored in the equation above...

        The pressure broadening parameters are summed from the contribution of
        each individual perturber so that for i perturbers

        .. math::
            \\gamma_pp = \\sum_i \\gamma_{p,i} p_i

        and

        .. math::
            \\delta f_pp = \\sum_i \\delta f_{p,i} p_i

        Parameters:
            temperature (float or ndarray): Temperature [Kelvin]

            line_temperature (float): Line temperature [Kelvin]

            pressure (float or like temperature): Total pressure [Pascal]

            vmrs (dict):  Volume mixing ratio of atmospheric species.
            dict should be {'self': self_vmr} for 'N2', {'self': self_vmr,
            'H2O': h2o_vmr} for kind 'WA', and each species of 'AP' should be
            represented in the same manner.  When 'self' is one of the list of
            species, then vmrs['self'] should not exist.  Missing data is
            treated as a missing species.  No data at all is assumed to mean
            1.0 VMR of self (len(vmrs) == 0 must evaluate as True).  The
            internal self_vmr, h2o_vmr, etc., variables must have sime size
            as pressure or be constants

        Returns:
            p · gamma0_p, p · delta-f0
        """
        theta = line_temperature / temperature
        if len(vmrs) == 0:
            return self._sgam * theta ** self._sn * pressure, \
                self._sdel * theta ** (0.25 + 1.5 * self._sn) * pressure

        sum_vmrs = 0.0
        gamma = np.zeros_like(temperature)
        delta_f = np.zeros_like(temperature)

        if self._type is self._none:
            return np.zeros_like(temperature), np.zeros_like(temperature)
        elif self._type is self._air:
            for species in vmrs:
                if species == 'self':
                    gamma += self._sgam * theta ** self._sn * \
                        pressure * vmrs[species]
                    delta_f += self._sdel * \
                        theta ** (0.25 + 1.5 * self._sn) * \
                        pressure * vmrs[species]
                    sum_vmrs += vmrs[species]
            gamma += self._agam * theta ** self._an * \
                pressure * (1 - sum_vmrs)
            delta_f += self._adel * theta ** (0.25 + 1.5 * self._an) * \
                pressure * (1 - sum_vmrs)
        elif self._type is self._air_and_water:
            for species in vmrs:
                if species == 'self':
                    gamma += self._sgam * theta ** self._sn * \
                        pressure * vmrs[species]
                    delta_f += self._sdel * \
                        theta ** (0.25 + 1.5 * self._sn) * \
                        pressure * vmrs[species]
                    sum_vmrs += vmrs[species]
                elif species == 'H2O':
                    gamma += self._wgam * theta ** self._wn * \
                        pressure * vmrs[species]
                    delta_f += self._wdel * \
                        theta ** (0.25 + 1.5 * self._wn) * \
                        pressure * vmrs[species]
                    sum_vmrs += vmrs[species]
            gamma += self._agam * theta ** self._an * \
                pressure * (1 - sum_vmrs)
            delta_f += self._adel * theta ** (0.25 + 1.5 * self._an) * \
                pressure * (1 - sum_vmrs)
        elif self._type is self._all_planets:
            for species in vmrs:
                if species == 'self':
                    gamma += self._sgam * theta ** self._sn * \
                        pressure * vmrs[species]
                    delta_f += self._sdel * \
                        theta ** (0.25 + 1.5 * self._sn) * \
                        pressure * vmrs[species]
                    sum_vmrs += vmrs[species]
                elif species in self._gam:
                    gamma += self._gam[species] * theta ** self._n[species] * \
                        pressure * vmrs[species]
                    delta_f += self._del[species] * \
                        theta ** (0.25 + 1.5 * self._n[species]) * \
                        pressure * vmrs[species]
                    sum_vmrs += vmrs[species]
            gamma /= sum_vmrs
            delta_f /= sum_vmrs
        return gamma, delta_f
