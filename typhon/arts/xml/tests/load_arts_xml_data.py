# -*- encoding: utf-8 -*-

import os
import fnmatch

from typhon.arts import xml


# Try to load all XML files in ARTS_DATA_PATH.
#
# Search for XML files in ARTS_DATA_PATH. If files are found, try to load
# them. It is just checked, if xml.load runs without exception.
#
# Notes:
#   This is not a docstring to ensure readble output in nosetests.
def test_load_arts_xml_data():
    if os.environ.get('ARTS_DATA_PATH'):
        for root, _, filenames in os.walk(os.environ.get('ARTS_DATA_PATH')):
            for filename in filenames:
                if filename.endswith(('.xml', '.xml.gz')):
                    yield _load_xml, os.path.join(root, filename)
    else:
        raise Exception('ARTS_DATA_PATH is not set.')

def _load_xml(f):
    """Load a given XML file."""
    xml.load(f)
    pass
