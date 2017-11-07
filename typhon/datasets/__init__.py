"""Modules in this package contain classes to handle datasets.

That includes the overall framework to handle datasets in the dataset
module, as well as concrete datasets for specific sensors etc., including
reading routines.

To implement a new reading routine, subclass one of the datasets here.
"""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

from . import dataset
