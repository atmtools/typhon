# -*- encoding: utf-8 -*-

import os

from typhon.arts import xml


# Try to load all XML files in ARTS_DATA_PATH.
#
# Search for XML files in ARTS_DATA_PATH. If files are found, try to load
# them. It is just checked, if xml.load runs without exception.
#
# Notes:
#   This is not a docstring to ensure readble output in nosetests.
data_path = os.getenv('ARTS_DATA_PATH')
def test_load_arts_xml_data():
    if data_path:
        for d in data_path.split(os.path.pathsep):
            for root, _, filenames in os.walk(d):
                for filename in filenames:
                    if filename.endswith(('.xml', '.xml.gz')):
                        yield _load_xml, os.path.join(root, filename)
    else:
        raise Exception('ARTS_DATA_PATH is not set.')

def _load_xml(f):
    """Load a given XML file."""
    xml.load(f)
    pass
