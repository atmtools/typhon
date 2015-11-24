# -*- coding: utf-8 -*-

__all__ = ['dimension_names', 'tensor_names', 'basic_types']

# Source: ARTS developer guide, section 3.4
dimension_names = [
    'ncols', 'nrows', 'npages', 'nbooks', 'nshelves', 'nvitrines', 'nlibraries']

tensor_names = [
    'Vector', 'Matrix', 'Tensor3', 'Tensor4', 'Tensor5', 'Tensor6', 'Tensor7']

basic_types = {
    'tuple': 'Array',
    'list': 'Array',
    'int': 'Index',
    'float': 'Numeric',
    'str': 'String',
}
