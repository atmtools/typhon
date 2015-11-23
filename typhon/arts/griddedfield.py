# -*- coding: utf-8 -*-

import numpy as np

__all__ = ['GriddedField1',
           'GriddedField2',
           'GriddedField3',
           'GriddedField4',
           'GriddedField5',
           'GriddedField6',
           'GriddedField7',
           ]


class GriddedField(object):
    """GriddedField simulates the behaviour of the same-named ARTS dataype.

    This class provides the facility of storing gridded data. For this purpose
    the grid-axes as well as the data are stored. GriddedFields can be easily
    written to XML-files as they define a clear datstructure.

    """
    def __init__(self, dim, grids=None, data=None, gridnames=None,
                 name=None):
        """Create a GriddedField object.

        Args:
            dim (int): Dimension of the GriddedField.
            grids (list, tuple, np.ndarray): grids.
            data (np.ndarray): data values.
            gridnames (List[str]): clear names for all grids.
            name (str): name of the GriddedField.

        """
        self._dimension = dim
        self._grids = grids
        self._data = data
        self._gridnames = gridnames
        self._name = name

    @property
    def dimension(self):
        return self._dimension

    @property
    def grids(self):
        return self._grids

    @property
    def gridnames(self):
        return self._gridnames

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    @grids.setter
    def grids(self, grids):
        """Set the grids of the GriddedField.

        This function sets the grids inside the GriddedField.

        Note:
            The number of grids has to match the GriddedField dimension.

        Args:
            grids (List[np.ndarray]): list of grids

        """
        if type(grids) not in (list, tuple):
            raise TypeError('The array of grids must be type list or tuple.')

        for grid in grids:
            if type(grid) not in (list, tuple, np.ndarray):
                raise TypeError('Grid must be of type list, tuple or np.ndarray')
            if (type(grid) in (list, tuple) and len(grid) > 0
                    and type(grid[0]) not in (str, int)):
                raise TypeError(('Grid must be of type list[int or str], '
                                 'tuple[int or str] or np.ndarray'))

        self._grids = grids

    @gridnames.setter
    def gridnames(self, gridnames):
        """Set the gridnames.

        This functions sets the names for the grids stored in the GriddedField.


        Note:
            The number of gridnames has to match the number of grids.
            Gridnames are currently not used so it is not neccesarry
            to set them.

        Args:
            gridnames (List[str]): list of gridnames

        """
        if all(isinstance(entry, str) for entry in gridnames):
            self._gridnames = gridnames
        else:
            raise TypeError('gridnames have to be type string')

    @data.setter
    def data(self, data):
        """Set the data array of the GriddedField.

        This function sets the data array for the GriddedField.

        Note:
            The data array has to fit the grid dimensions.

        Args:
            data (np.ndarray): data array

        """
        self._data = data

    @name.setter
    def name(self, name):
        """Set the name of the GriddedField.

        This function sets name of the GriddedField.

        Args:
            name (str): name of the GriddedField

        """
        if type(name) is str:
            self._name = name
        else:
            raise TypeError('Name has to be type string')

    def check_dimension(self):
        """Checks the consistency of grids and data.

        This functions check if the dimensions defined by the grids fit to the
        dimension of the passed data.
        Also check if the number of gridnames fits the number of grids.

        Note:
            The check has to be done manually after the GriddedField is filled
            completely. The check is not run automatically while setting the
            data array.

        Returns:
            True if successful.

        Raises:
            Exception: if number of grids does not fit
                the GriddedField dimension.
            Exception: if number of gridnames does not fit
                the number of grids.
            Exception: if data dimension does not fit the grid dimensions.
            Warning: if a dimension is empty.

        """
        # define error messages
        grid_dim_error = (('The number of grids has to fit the dimension '
                           'of the GriddedField.\nThe dimension is {0} '
                           'but {1} grids were passed.')
                           .format(self.dimension, len(self.grids)))

        grid_name_error = (('The number of gridnames has to fit the '
                            'dimension of the GriddedField.\nThe dimension'
                            ' is {0} but {1} gridnames were passed.')
                            .format(self.dimension, len(self.gridnames)))

        # number of grids has to match the GriddedField dimension
        if len(self.grids) != self.dimension:
            raise Exception(grid_dim_error)

        # if grids are named, each grid has to be named
        if self.gridnames is None:
            self.gridnames = [''] * self.dimension
        else:
            if len(self.gridnames) != self.dimension:
                raise Exception(grid_name_error)

        # grid and data dimension have to fit
        g_dim = [np.size(g) if np.size(g) > 0 else 1 for g in self.grids]

        if tuple(g_dim) != self.data.shape:
            raise Exception(('Dimension mismatch between data and grids. '
                             'Grid dimension is {0} but data {1}')
                             .format(tuple(g_dim), self.data.shape))

        return True


class GriddedField1(GriddedField):
    """GriddedField with 1 dimensions."""
    def __init__(self):
        super(GriddedField1, self).__init__(1)


class GriddedField2(GriddedField):
    """GriddedField with 2 dimensions."""
    def __init__(self):
        super(GriddedField2, self).__init__(2)


class GriddedField3(GriddedField):
    """GriddedField with 3 dimensions."""
    def __init__(self):
        super(GriddedField3, self).__init__(3)


class GriddedField4(GriddedField):
    """GriddedField with 4 dimensions."""
    def __init__(self):
        super(GriddedField4, self).__init__(4)


class GriddedField5(GriddedField):
    """GriddedField with 5 dimensions."""
    def __init__(self):
        super(GriddedField5, self).__init__(5)


class GriddedField6(GriddedField):
    """GriddedField with 6 dimensions."""
    def __init__(self):
        super(GriddedField6, self).__init__(6)


class GriddedField7(GriddedField):
    """GriddedField with 7 dimensions."""
    def __init__(self):
        super(GriddedField7, self).__init__(7)
