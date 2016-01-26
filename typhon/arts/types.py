# -*- coding: utf-8 -*-

"""Collection of all ARTS types."""

from .griddedfield import *
from .scattering import *
from .retrieval import *
from .catalogues import *

__all__ = []

classes = {
    'ArrayOfLineRecord': ArrayOfLineRecord,
    'CIARecord': CIARecord,
    'GasAbsLookup': GasAbsLookup,
    'GriddedField1': GriddedField1,
    'GriddedField2': GriddedField2,
    'GriddedField3': GriddedField3,
    'GriddedField4': GriddedField4,
    'GriddedField5': GriddedField5,
    'GriddedField6': GriddedField6,
    'GriddedField7': GriddedField7,
    'LineMixingRecord': LineMixingRecord,
    'QuantumIdentifier': QuantumIdentifier,
    'QuantumNumberRecord': QuantumNumberRecord,
    'QuantumNumbers': QuantumNumbers,
    'RetrievalQuantity': RetrievalQuantity,
    'ScatteringMetaData': ScatteringMetaData,
    'SingleScatteringData': SingleScatteringData,
    'Sparse': Sparse,
    'SpeciesAuxData': SpeciesAuxData,
    'SpeciesTag': SpeciesTag,
}
