"""Modules in this package contain classes to handle datasets.

That includes the overall framework to handle datasets in the dataset
module, as well as concrete datasets for specific sensors etc., including
reading routines.

To implement a new reading routine, subclass one of the datasets here.
"""

from . import dataset
