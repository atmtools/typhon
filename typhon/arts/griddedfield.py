# -*- coding: utf-8 -*-
import copy
import numbers

import numpy as np
from scipy import interpolate
import xarray

from .utils import return_if_arts_type, get_arts_typename

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
    """:class:`GriddedField` implements the same-named ARTS dataype.

    This class provides the facility of storing gridded data. For this purpose
    the grid-axes as well as the data are stored. GriddedFields can be easily
    written to XML-files as they define a clear datastructure.

    :class:`GriddedField` should not be used directly. Use one of the derived
    types such as :class:`GriddedField1` instead.

    Note:
        For the special case of storing atmospheric profiles as GriddedField3
        the latitude and longitude grids have to be initialised as empty
        np.array.

    Examples:
        Create and manipulate a :class:`GriddedField` object.

        >>> gf1 = GriddedField1()
        >>> gf1.grids = [np.arange(10)]
        >>> gf1.gridnames = ["Indices"]
        >>> gf1.data = np.random.randn(10)

        Inspect an existing :class:`GriddedField` object.

        >>> gf1.dimension
        1
        >>> gf1.grids
        [array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
        >>> gf1.gridnames
        ['Indices']

    """

    def __init__(self, dimension, grids=None, data=None, gridnames=None,
                 dataname=None, name=None):
        """Create a GriddedField object.

        Parameters:
            dimension (int): Dimension of the GriddedField.
            grids (list, tuple, np.ndarray): grids.
            data (np.ndarray): data values.
            gridnames (List[str]): clear names for all grids.
            dataname (str): name of the data array.
            name (str): name of the GriddedField.

        """
        self._dimension = return_if_arts_type(dimension, 'Index')
        self.grids = copy.deepcopy(grids)
        self.data = copy.copy(data)
        self.gridnames = copy.copy(gridnames)
        self.dataname = dataname
        self.name = name

    def __getitem__(self, index):
        """Make the data array subscriptable directly.

        ``gf[0, 1]`` is equivalent to ``gf.data[0, 1]``.
        """
        return self.data[index]

    def __eq__(self, other):
        """Test the equality of GriddedFields."""
        if isinstance(other, self.__class__):
            # Check each attribute after another for readabilty.
            # Return as soon as possible for performance.
            if self.name != other.name:
                return False

            if self.dataname != other.dataname:
                return False

            if self.gridnames != other.gridnames:
                return False

            if self.dimension != other.dimension:
                return False

            if self.grids is not None and other.grids is not None:
                if not np.all(a == b for a, b in zip(self.grids, other.grids)):
                    return False
            elif self.grids is not other.grids:
                return False

            if self.data is not None and other.data is not None:
                if not np.allclose(self.data, other.data):
                    return False
            elif self.data is not other.data:
                return False

            return True
        return NotImplemented

    def __neq__(self, other):
        """Test the non-equality of GriddedFields."""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __repr__(self):
        try:
            if self.name:
                out = "GriddedField{}: {}\n".format(self.dimension, self.name)
            else:
                out = "GriddedField{}: {}\n".format(self.dimension,
                                                    "Generic Field")

            for i in range(self.dimension):
                if self.gridnames[i]:
                    out += "{} {}: {}\n".format(self.gridnames[i],
                                                np.shape(self.grids[i]),
                                                self.grids[i])
                else:
                    out += "{} {}: {}\n".format("Grid " + str(i + 1),
                                                np.shape(self.grids[i]),
                                                self.grids[i])
            if self.dataname:
                out += "{} {}: {}\n".format(self.dataname,
                                            self.data.shape,
                                            self.data.flatten())
            else:
                out += "{} {}: {}\n".format("Data",
                                            self.data.shape,
                                            self.data.flatten())
            return out
        except:
            # If representation fails, fall back to default.
            # Known issues: Empty GriddedFields.
            return '<{0} at {1}>'.format(type(self).__name__, hex(id(self)))

    @property
    def shape(self):
        """Shape of the data array."""
        return self.data.shape

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
        """List of grids defining the GriddedField.

        Note:
            The number of grids has to match the GriddedField dimension.
        """
        return self._grids

    @property
    def gridnames(self):
        """A list or tuple that includes a name for every grid.

        Note:
            The number of gridnames has to match the number of grids.
            Gridnames are currently not used so it is not neccesarry
            to set them.
        """
        return self._gridnames

    @property
    def data(self):
        """The data matrix stored in the GriddedField.

        Note:
            The data array has to fit the grid dimensions.
        """
        return self._data

    @property
    def name(self):
        """Name of the GriddedField."""
        return self._name

    @property
    def dataname(self):
        """Name of the data array."""
        return self._dataname

    @grids.setter
    def grids(self, grids):
        if grids is None:
            self._grids = None
            return

        if type(grids) not in (list, tuple):
            raise TypeError('The array of grids must be type list or tuple.')

        for grid in grids:
            if (get_arts_typename(grid)
                    in ['ArrayOfString', 'ArrayOfIndex', 'Vector', None]):
                self._grids = grids
            else:
                raise TypeError(
                    'grids have to be ArrayOfString, ArrayOfIndex or Vector.')

    @gridnames.setter
    def gridnames(self, gridnames):
        self._gridnames = return_if_arts_type(gridnames, 'ArrayOfString')

    @data.setter
    def data(self, data):
        data_type = get_arts_typename(np.ndarray([0] * self.dimension))
        self._data = return_if_arts_type(data, data_type)

    @dataname.setter
    def dataname(self, dataname):
        self._dataname = return_if_arts_type(dataname, 'String')

    @name.setter
    def name(self, name):
        self._name = return_if_arts_type(name, 'String')

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

    def check_atm_fields_compact(self, check_gridnames=False):
        """Checks the consistency of grids and data.

        This functions check if the dimensions defined by the grids fit to the
        dimension of the passed data.
        Also check if the number of gridnames fits the number of grids.

        Note:
            The last three gridnames are expected to be 'Pressure', 'Latitude'
            and 'Longitude'. This is more strict than the ARTS checks, but good
            behavior.

        Returns:
            True if successful.

        Raises:
            Exception:
                - If not GriddedField4.
                - If the pressure grid is not decreasing.

        """
        if self.dimension != 4:
            raise Exception('atm_fields_compact have to be GriddedField4.')

        if check_gridnames:
            if self.gridnames[1] != 'Pressure':
                err = "Second grid has to be 'Pressure' not '{}'."
                raise Exception(err.format(self.gridnames[1]))
            elif self.gridnames[2] != 'Latitude':
                err = "Third grid has to be 'Latitude' not '{}'."
                raise Exception(err.format(self.gridnames[2]))
            elif self.gridnames[3] != 'Longitude':
                err = "Fourth grid has to be 'Longitude' not '{}'."
                raise Exception(err.format(self.gridnames[3]))

        if not get_arts_typename(self.grids[0]) == 'ArrayOfString':
            raise Exception('First grid has to be ArrayOfString.')

        if not all(np.diff(self.grids[1]) < 0):
            raise Exception('Pressure grid has to be strictly decreasing.')

        return True

    def copy(self):
        """Return a deepcopy of the GriddedField."""
        return copy.deepcopy(self)

    def extract_slice(self, s=slice(None), axis=0):
        """Return a new GriddedField containing a slice of the current one.

        Parameters:
            s (slice): Slice.
            axis (int): Axis to slice along.

        Returns:
            :class:`typhon.arts.griddedfield.GriddedField`:
                GriddedField containing sliced grids and data.
        """
        gf = self.copy()
        gf.grids[axis] = gf.grids[axis][s]
        slices = [slice(None) for i in range(self.dimension)]
        slices[axis] = s
        gf.data = gf.data[slices]

        return gf

    def refine_grid(self, new_grid, axis=0, fun=np.array, **kwargs):
        """Interpolate GriddedField axis to a new grid.

        This function replaces a grid of a GriddField and interpolates all
        data to match the new coordinates. :func:`scipy.interpolate.interp1d`
        is used for interpolation.

        Parameters:
            new_grid (ndarray): The coordinates of the interpolated values.
            axis (int): Specifies the axis of data along which to interpolate.
                Interpolation defaults to the first axis of the GriddedField.
            fun (numpy.ufunc, or similar): Function to apply to grid before
                interpolation.  Suggested values: np.array, np.log10, np.log
            **kwargs:
                Keyword arguments passed to :func:`scipy.interpolate.interp1d`.

        Returns: :class:`typhon.arts.griddedfield.GriddedField`

        """
        if len(self.grids[axis]) > 1:
            f = interpolate.interp1d(fun(self.grids[axis]), self.data,
                                     axis=axis, **kwargs)
            self.grids[axis] = new_grid
            self.data = f(fun(new_grid))
        else:  # if the intention is to create a useful TensorX
            self.data = self.data.repeat(len(new_grid), axis=axis)
            self.grids[axis] = new_grid

        self.check_dimension()

        return self

    def get(self, key, default=None, keep_dims=True):
        """Return data from field with given fieldname.

        Notes:
              This method only works, if the first grid
              is an :arts:`ArrayOfString`.

        Parameters:
              key (str): Name of the field to extract.
              default: Default value, if ``key`` is not found.
              keep_dims (bool): If ``False``, empty dimensions are squeezed
                  before the extracted array is returned.

        Returns:
            ndarray: Extracted ndarray.
        """
        # The first grid has to be an ArrayOfString.
        if not get_arts_typename(self.grids[0]) == 'ArrayOfString':
            raise TypeError(
                'Method only works, if the first grid is an "ArrayOfString"')

        # If the GriddedField is empty or the given fieldname is not found,
        # return the default value.
        if self.grids is None or key not in self.grids[0]:
            return default

        # Find the index of given fieldname in the name grid and return the
        # ndarray at that position.
        field = self.data[[self.grids[0].index(key)]]

        # Squeeze empty dimensions, if ``keep_dims`` is ``False``.
        return field if keep_dims else field.squeeze()

    def set(self, key, data):
        """Assign data to field with given fieldname.

        Notes:
              This method only works, if the first grid
              is an :arts:`ArrayOfString`.

        Parameters:
              key (str): Name of the field to extract.
              data (ndarray): Data array.
        """
        if not get_arts_typename(self.grids[0]) == 'ArrayOfString':
            raise TypeError(
                'Method only works, if the first grid is an "ArrayOfString"')

        self.data[[self.grids[0].index(key)]] = data

    def scale(self, key, factor, dtype=float):
        """Scale data stored in field with given fieldname.

        Notes:
              This method only works, if the first grid
              is an :arts:`ArrayOfString`.

        Parameters:
              key (str): Name of the field to scale.
              data (float or ndarray): Scale factor.
              dtype (type): Data type used for typecasting. If the original
                dtype of ``GriddedField.data`` is ``int``, the data array
                gets typecasted to prevent messy behaviour when assigning
                scaled values.
        """
        if issubclass(self.data.dtype.type, numbers.Integral):
            # Typecast integer data arrays to prevent unwanted typecast when
            # assigning scaled (float) variables back to the (integer) ndarray.
            self.data = self.data.astype(dtype)

        self.set(key, self.get(key) * factor)

    def to_dict(self):
        """Convert GriddedField to dictionary.

        Converts a GriddedField object into a classic Python dictionary. The
        gridname is used as dictionary key. If the grid is unnamed the key is
        generated automatically ('grid1', 'grid2', ...). The data can be
        accessed through the 'data' key.

        Returns:
            Dictionary containing the grids and data.

        """
        grids, gridnames = self.grids, self.gridnames

        if gridnames is None:
            gridnames = ['grid%d' % n for n in range(1, self.dimension + 1)]

        for n, name in enumerate(gridnames):
            if name == '':
                gridnames[n] = 'grid%d' % (n + 1)

        d = {name: grid for name, grid in zip(gridnames, grids)}

        if self.dataname is not None:
            d[self.dataname] = self.data
        else:
            d['data'] = self.data

        return d

    def to_xarray(self):
        """Convert GriddedField to xarray.DataArray object.

        Convert a GriddedField object into a :func:`xarray.DataArray`
        object.  The dataname is used as the DataArray name.

        Returns:
            xarray.DataArray object corresponding to gridded field
        """

        da = xarray.DataArray(self.data)
        da = da.rename(dict((k, v)
            for (k, v) in zip(da.dims, self.gridnames)
            if v!=""))
        da = da.assign_coords(
            **{name: coor
                for (name, coor) in zip(da.dims, self.grids)
                if len(coor)>0})
        if self.name is not None:
            da.name = self.name
        return da

    @classmethod
    def from_nc(cls, inputfile, variable, fill_value=np.nan):
        """Create GriddedField from variable in netCDF files.

        Extract a given variable from a netCDF file. The data and its
        dimensions are returned as a :class:`GriddedField` object.

        Parameters:
            inputfile (str): Path to netCDF file.
            variable (str): Variable key of variable to extract.
            fill_value (float): Fill value for masked areas (default: np.nan).

        Returns:
            GriddedField object of sufficient dimension.

        Raises:
            Exception: If the variable key can't be found in the netCDF file.

        """
        from netCDF4 import Dataset
        nc = Dataset(inputfile)

        if variable not in nc.variables:
            raise Exception('netCDF file has no variable {}.'.format(variable))

        data = nc.variables[variable]

        obj = cls(data.ndim)
        obj.grids = [nc.variables[dim][:] for dim in data.dimensions]
        obj.gridnames = [dim for dim in data.dimensions]

        if isinstance(data[:], np.ma.MaskedArray):
            obj.data = data[:].filled(fill_value=fill_value)
        else:
            obj.data = data[:]

        obj.check_dimension()

        return obj

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

        # Read data (and optional dataname).
        obj.data = xmlelement[-1].value()
        if 'name' in xmlelement[-1].attrib:
            obj.dataname = xmlelement[-1].attrib['name']

        obj.check_dimension()
        return obj

    def to_atmlab_dict(self):
        """Returns a copy of the GriddedField as a dictionary.

        Returns a dictionary compatible with an atmlab structure.

        Returns:
            Dictionary containing the grids and data.
        """

        d = {}
        if self.name is None:
            d['name'] = ''
        else:
            d['name'] = self.name
        d['grids'] = self.grids
        d['gridnames'] = self.gridnames
        d['data'] = self.data
        if self.dataname is None:
            d['dataname'] = ''
        else:
            d['dataname'] = self.dataname

        return d

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

        if self.dataname is None:
            xmlwriter.write_xml(self.data)
        else:
            xmlwriter.write_xml(self.data, {'name': self.dataname})

        xmlwriter.close_tag()


class GriddedField1(GriddedField):
    """GriddedField with 1 dimension."""

    def __init__(self, *args, **kwargs):
        super(GriddedField1, self).__init__(1, *args, **kwargs)


class GriddedField2(GriddedField):
    """GriddedField with 2 dimensions."""

    def __init__(self, *args, **kwargs):
        super(GriddedField2, self).__init__(2, *args, **kwargs)


class GriddedField3(GriddedField):
    """GriddedField with 3 dimensions."""

    def __init__(self, *args, **kwargs):
        super(GriddedField3, self).__init__(3, *args, **kwargs)


class GriddedField4(GriddedField):
    """GriddedField with 4 dimensions."""

    def __init__(self, *args, **kwargs):
        super(GriddedField4, self).__init__(4, *args, **kwargs)


class GriddedField5(GriddedField):
    """GriddedField with 5 dimensions."""

    def __init__(self, *args, **kwargs):
        super(GriddedField5, self).__init__(5, *args, **kwargs)


class GriddedField6(GriddedField):
    """GriddedField with 6 dimensions."""

    def __init__(self, *args, **kwargs):
        super(GriddedField6, self).__init__(6, *args, **kwargs)


class GriddedField7(GriddedField):
    """GriddedField with 7 dimensions."""

    def __init__(self, *args, **kwargs):
        super(GriddedField7, self).__init__(7, *args, **kwargs)
