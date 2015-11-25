# -*- encoding: utf-8 -*-
#!/usr/bin/env python

import numpy as np

from nose.tools import raises

from typhon.arts import griddedfield

class TestGriddedFieldUsage():
    def test_check_init(self):
        """Test initialisation of GriddedFields."""
        for dim in np.arange(1,8):
            gf = griddedfield.GriddedField(dim)
            assert gf.dimension == dim

    def test_check_dimension1(self):
        """Test if grid and data dimension agree (positive)."""
        gf3 = griddedfield.GriddedField3()
        gf3.grids = [np.arange(5), np.arange(5), []]
        gf3.gridnames = ["A", "B", "C"]
        gf3.data = np.ones((5, 5, 1))
        assert gf3.check_dimension() is True

    @raises(Exception)
    def test_check_dimension2(self):
        """Test if grid and data dimension agree (negative)."""
        gf3 = griddedfield.GriddedField3()
        gf3.grids = [np.arange(5), np.arange(5), []]
        gf3.gridnames = ["A", "B", "C"]
        gf3.data = np.ones((5, 5))
        gf3.check_dimension()

    def test_data(self):
        """Test setting and getting of data. """
        reference = np.random.randn(10, 10)
        gf3 = griddedfield.GriddedField3()
        gf3.data = reference
        assert np.array_equal(gf3.data, reference)

    def test_name_setter(self):
        """Test name setter and getter."""
        reference = 'TestName'
        gf = griddedfield.GriddedField1()
        gf.name = reference
        assert gf.name == reference

    def test_name_type(self):
        """Test if only names of type str are accepted."""
        for false_type in [float(), int(), tuple()]:
            yield self._set_name_of_type, false_type

    @raises(TypeError)
    def _set_name_of_type(self, name_type):
        gf = griddedfield.GriddedField1()
        gf.name = name_type


class TestGriddedFieldLoad():
    def test_load(self):
        """Load reference XML file for all GriddedField types."""
        pass


class TestGriddedFieldWrite():
    def test_write(self):
        """Save GriddedField to XML file, read it and compare the results."""
        pass
