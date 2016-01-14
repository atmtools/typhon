# -*- coding: utf-8 -*-

import numpy as np

__all__ = ['GriddedField',
           'GriddedField1',
           'GriddedField2',
           'GriddedField3',
           'GriddedField4',
           'GriddedField5',
           'GriddedField6',
           'GriddedField7',
           ]


class GriddedField(object):
    """:class:`GriddedField` simulates the behaviour of the same-named ARTS dataype.

    This class provides the facility of storing gridded data. For this purpose
    the grid-axes as well as the data are stored. GriddedFields can be easily
    written to XML-files as they define a clear datastructure.

    :class:`GriddedField` should not be used directly. Use one of the derived
    types such as :class:`GriddedField1` instead.

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
        if np.mod(dim, 1) == 0:
            self._dimension = int(dim)
        else:
            raise TypeError('Dimension has to be integer.')

        self.grids = grids
        self.data = data
        self.gridnames = gridnames
        self.name = name

    @property
    def dimension(self):
        """Dimension of the GriddedField.

        The dimension has to be defined when creating the GriddedField object.
        For the convenience subclasses (e.g. GriddedField1) this is done
        automatically.

        """
        return self._dimension

    @property
    def grids(self):
        """List of grids defining the GriddedField."""
        return self._grids

    @property
    def gridnames(self):
        """A list or tuple that includes a name for every grid."""
        return self._gridnames

    @property
    def data(self):
        """The data matrix stored in the GriddedField."""
        return self._data

    @property
    def name(self):
        """Name of the GriddedField."""
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
        if grids is None:
            self._grids = None
            return

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
        if gridnames is None:
            self._gridnames = None
            return

        if type(gridnames) not in (list, tuple):
            raise TypeError('The array of gridnames must be type list or tuple.')

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
        if data is None:
            self._data = None
            return

        if type(data) is np.ndarray:
            self._data = data
        else:
            raise TypeError('Data has to be np.ndarray.')

    @name.setter
    def name(self, name):
        """Set the name of the GriddedField.

        This function sets name of the GriddedField.

        Args:
            name (str): name of the GriddedField

        """
        if name is None:
            self._name = None
            return

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
            This check is done automatically before storing and after loading
            XML files.

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

        # number of grids has to match the GriddedField dimension
        if len(self.grids) != self.dimension:
            raise Exception(grid_dim_error)

        # if grids are named, each grid has to be named
        if self.gridnames is None:
            self.gridnames = [''] * self.dimension

        grid_name_error = (('The number of gridnames has to fit the '
                            'dimension of the GriddedField.\nThe dimension'
                            ' is {0} but {1} gridnames were passed.')
                            .format(self.dimension, len(self.gridnames)))

        if len(self.gridnames) != self.dimension:
            raise Exception(grid_name_error)

        # grid and data dimension have to fit
        g_dim = [np.size(g) if np.size(g) > 0 else 1 for g in self.grids]

        if tuple(g_dim) != self.data.shape:
            raise Exception(('Dimension mismatch between data and grids. '
                             'Grid dimension is {0} but data {1}')
                             .format(tuple(g_dim), self.data.shape))

        return True

    @classmethod
    def from_xml(cls, xmlelement):
        """Load a GriddedField from an ARTS XML file.

        Returns:
            GriddedField. Dimension depends on data in file.


        """
        obj = cls()

        if 'name' in xmlelement.attrib:
            obj.name = xmlelement.attrib['name']

        obj.grids = [x.value() for x in xmlelement[:-1]]
        obj.gridnames = [x.attrib['name']
                         if 'name' in x.attrib else ''
                         for x in xmlelement[:-1]]

        obj.data = xmlelement[-1].value()

        obj.check_dimension()
        return obj

    def write_xml(self, xmlwriter, attr=None):
        """Save a GriddedField to an ARTS XML file."""
        self.check_dimension()

        if attr is None:
            attr = {}

        if self.name is not None:
            attr['name'] = self.name

        xmlwriter.open_tag('GriddedField{}'.format(self.dimension), attr)
        for grid, name in zip(self.grids, self.gridnames):
            xmlwriter.write_xml(grid, {'name': name})

        xmlwriter.write_xml(self.data, {'name': 'Data'})

        xmlwriter.close_tag()


class GriddedField1(GriddedField):
    """GriddedField with 1 dimension."""
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
