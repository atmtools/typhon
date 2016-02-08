# -*- encoding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import os
from tempfile import mkstemp

from nose.tools import raises

from typhon.arts import griddedfield, xml


def _create_tensor(n):
    """Create a tensor of dimension n.

    Create a tensor with n dimensions with two entries in each dimension.
    The tensor is filled with increasing integers starting with 0.

    Args:
        n (int): number of dimensions

    Returns:
        np.ndarray: n-dimensional tensor

    """
    return np.arange(2 ** n).reshape(2 * np.ones(n).astype(int))


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
        reference = np.random.randn(10, 10, 10)
        gf3 = griddedfield.GriddedField3()
        gf3.data = reference
        assert np.array_equal(gf3.data, reference)


    def test_name_setter(self):
        """Test name setter and getter."""
        reference = 'TestName'
        gf = griddedfield.GriddedField1()
        gf.name = reference
        assert gf.name == reference


    def test_to_dict(self):
        """Test the conversion into a dictionary."""
        gf2 = griddedfield.GriddedField2()
        gf2.grids = [np.ones(5), np.zeros(5)]
        gf2.gridnames = ["ones", "zeros"]
        gf2.data = np.ones((5, 5))
        d = gf2.to_dict()

        res = (np.array_equal(d['ones'], np.ones(5))
            and np.array_equal(d['zeros'], np.zeros(5))
            and np.array_equal(d['data'], np.ones((5, 5))))

        assert res is True

    def test_name_type(self):
        """Test if only names of type str are accepted."""
        for false_type in [float(), int()]:
            yield self._set_name_of_type, false_type


    @raises(TypeError)
    def _set_name_of_type(self, name_type):
        gf = griddedfield.GriddedField1()
        gf.name = name_type


class TestGriddedFieldLoad():
    ref_dir = os.path.join(os.path.dirname(__file__), "reference", "")


    def test_load_data(self):
        """Load reference XML file for GriddedField3 and check the data."""
        reference = _create_tensor(3)
        gf3 = xml.load(self.ref_dir + 'GriddedField3.xml')
        test_data = gf3.data
        assert np.array_equal(test_data, reference)


    def test_load_grids(self):
        """Load reference XML file for GriddedField3 and check the grids."""
        reference = [np.arange(2)] * 3
        gf3  = xml.load(self.ref_dir + 'GriddedField3.xml')
        test_data = gf3.grids
        assert all(np.allclose(a, b) for a, b in zip(test_data, reference))


    def test_load_gridnames(self):
        """Load reference XML file for GriddedField3 and check the gridnames."""
        reference = ['grid1', 'grid2', 'grid3']
        gf3 = xml.load(self.ref_dir + 'GriddedField3.xml')
        test_data = gf3.gridnames
        assert np.array_equal(test_data, reference)


    def test_load_dimension(self):
        """Load reference XML file for GriddedField3 and do the dimension check."""
        gf3 = xml.load(self.ref_dir + 'GriddedField3.xml')
        assert gf3.check_dimension()


class TestGriddedFieldWrite():
    def setUp(self):
        """Create a temporary file."""
        _, self.f = mkstemp()
        print(self.f)


    def tearDown(self):
        """Delete temporary file."""
        os.remove(self.f)


    def test_write(self):
        """Save GriddedField to XML file, read it and compare the results."""
        for dim in np.arange(1, 8):
            yield self._load_griddedfield, dim


    def _load_griddedfield(self, dim):
        gf = griddedfield.GriddedField(dim)
        gf.grids = [np.arange(2)] * dim
        gf.data = _create_tensor(dim)
        xml.save(gf, self.f)
        test_data = xml.load(self.f)
        assert np.array_equal(gf.data, test_data.data)
