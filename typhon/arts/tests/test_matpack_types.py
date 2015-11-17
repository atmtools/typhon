# -*- coding: utf-8 -*-

"""Testing the basic ARATS XML functions

This module provides basic functions to test the reading and writing
of ARTS XML files.
"""

import os
import numpy as np
from nose.tools import with_setup
from typhon.arts import xml

# set path to reference directory as global variable
ref_dir = os.path.join(os.path.dirname(__file__), "reference", "")


def test_load_index():
    """Load reference XML file for ARTS type Index."""
    assert xml.load(ref_dir + 'index.xml') == 0


def test_load_vector():
    """Load reference XML file for ARTS type Vector."""
    reference = _create_tensor(1)
    test_data = xml.load(ref_dir + 'vector.xml')
    assert np.array_equal(test_data, reference)


def test_load_matrix():
    """Load reference XML file for ARTS type Matrix."""
    reference = _create_tensor(2)
    test_data = xml.load(ref_dir + 'matrix.xml')
    assert np.array_equal(test_data, reference)


def test_load_tensor():
    """Load reference XML files for different tensor types."""
    for n in range(3, 8):
        yield _load_tensor, n


def test_load_arrayofindex():
    """Load reference XML file for ARTS type ArrayOfIndex."""
    reference = [1., 2., 3.]
    test_data = xml.load(ref_dir + 'arrayofindex.xml')
    assert np.array_equal(test_data, reference)


def test_load_arrayofstring():
    """Load reference XML file for ARTS type ArrayOfString."""
    reference = ['a', 'bb', 'ccc']
    test_data = xml.load(ref_dir + 'arrayofstring.xml')
    assert np.array_equal(test_data, reference)


def _load_tensor(n):
    """Load tensor of dimension n and compare data to reference.

    Args:
        n (int): number of dimensions

    """
    reference = _create_tensor(n)
    test_data = xml.load(ref_dir + 'tensor{}.xml'.format(n))
    assert np.array_equal(test_data, reference)


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
