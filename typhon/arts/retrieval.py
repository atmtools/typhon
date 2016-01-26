# -*- coding: utf-8 -*-
"""
Implementation of RetrievalQuantity.

"""

import numpy as np

__all__ = ['RetrievalQuantity',
           ]


class RetrievalQuantity:
    """Represents a RetrievalQuantity object.

    See online ARTS documentation for object details.

    """

    def __init__(self, maintag=None, subtag=None, subsubtag=None, mode=None,
            analytical=None, pertubation=None, grids=None):

        self.maintag = maintag
        self.subtag = subtag
        self.subsubtag = subsubtag
        self.mode = mode
        self.analytical = analytical
        self.pertubation = pertubation
        self.grids = grids

    @property
    def maintag(self):
        """MainTag of retrieval species."""
        return self._maintag

    @property
    def subtag(self):
        """Subtag of retrieval species."""
        return self._subtag

    @property
    def subsubtag(self):
        """Subsubtag of retrieval species."""
        return self._subsubtag

    @property
    def mode(self):
        """Retrieval mode."""
        return self._mode

    @property
    def analytical(self):
        """Flag to determine whether the retrieval was done analytically."""
        return self._analytical

    @property
    def pertubation(self):
        """Amplitude of the pertubation."""
        return self._pertubation

    @property
    def grids(self):
        """Pressure grid."""
        return self._pertubation

    @maintag.setter
    def maintag(self, maintag):
        if maintag is None:
            self._maintag = None
            return

        if type(maintag) is str:
            self._maintag = maintag
        else:
            raise TypeError('maintag has to be str.')

    @subtag.setter
    def subtag(self, subtag):
        if subtag is None:
            self._subtag = None
            return

        if type(subtag) is str:
            self._subtag = subtag
        else:
            raise TypeError('subtag has to be str.')

    @subsubtag.setter
    def subsubtag(self, subsubtag):
        if subsubtag is None:
            self._subsubtag = None
            return

        if type(subsubtag) is str:
            self._subsubtag = subsubtag
        else:
            raise TypeError('subsubtag has to be str.')

    @mode.setter
    def mode(self, mode):
        if mode is None:
            self._mode = None
            return

        if type(mode) is str:
            self._mode = mode
        else:
            raise TypeError('mode has to be str.')

    @analytical.setter
    def analytical(self, analytical):
        if analytical is None:
            self._analytical = None
            return

        self._analytical = analytical

    @pertubation.setter
    def pertubation(self, pertubation):
        if pertubation is None:
            self._pertubation = None
            return

        self._pertubation = pertubation

    @grids.setter
    def grids(self, grids):
        if grids is None:
            self._grids = None
            return

        if type(grids) is list and type(grids[0]) is np.ndarray:
            self._grids = grids
        else:
            raise TypeError('grids has to be list of np.ndarray.')

    @classmethod
    def from_xml(cls, xmlelement):
        """Loads a RetrievalQuantity object from an existing file.

        """
        obj = cls()
        obj.maintag = xmlelement[0].value()
        obj.subtag = xmlelement[1].value()
        obj.subsubtag = xmlelement[2].value()
        obj.mode = xmlelement[3].value()
        obj.analytical = xmlelement[4].value()
        obj.pertubation = xmlelement[5].value()
        obj.grids = xmlelement[6].value()

        return obj

    def write_xml(self, xmlwriter, attr=None):
        """Write a RetrievalQuantity object to an ARTS XML file.
        """
        if attr is None:
            attr = {}

        xmlwriter.open_tag("RetrievalQuantity", attr)
        xmlwriter.write_xml(self.maintag, {'name': 'MainTag'})
        xmlwriter.write_xml(self.subtag, {'name': 'Subtag'})
        xmlwriter.write_xml(self.subsubtag, {'name': 'SubSubtag'})
        xmlwriter.write_xml(self.mode, {'name': 'Mode'})
        xmlwriter.write_xml(self.analytical, {'name': 'Analytical'})
        xmlwriter.write_xml(self.pertubation, {'name': 'Perturbation'})
        xmlwriter.write_xml(self.grids, {'name': 'Grids'})
        xmlwriter.close_tag()

