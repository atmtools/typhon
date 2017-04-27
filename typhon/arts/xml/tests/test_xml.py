# -*- coding: utf-8 -*-
"""Testing high-level functionality in typhon.arts.xml.
"""
import numpy as np
from nose.tools import raises

from typhon.arts import xml


class TestXML(object):
    """Testing high-level functionality in typhon.arts.xml."""
    def test_load_directory(self):
        """Test loading all XML files in a directory."""
        t = xml.load_directory('reference/')
        ref = xml.load('reference/vector.xml')

        assert np.allclose(t['vector'], ref)

    @raises(KeyError)
    def test_load_directory_exclude(self):
        """Test excluding files when loading directory content."""
        t = xml.load_directory('reference/', exclude=['vector.xml'])
        t['vector']
