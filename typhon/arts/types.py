# -*- coding: utf-8 -*-

"""Collection of all ARTS types."""

from .griddedfield import *
from .scattering import *

__all__ = []

classes = {
    'GriddedField1': GriddedField1,
    'GriddedField2': GriddedField2,
    'GriddedField3': GriddedField3,
    'GriddedField4': GriddedField4,
    'GriddedField5': GriddedField5,
    'GriddedField6': GriddedField6,
    'GriddedField7': GriddedField7,
    'SingleScatteringData': SingleScatteringData,
    'ScatteringMetaData': ScatteringMetaData,
}
