"""Contains classes that extend the functionality of plain numpy ndarrays,
to bundle them in labeled groups and store them to netCDF4 files.
"""

import copy
from datetime import datetime
from itertools import chain
import textwrap
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

try:
    import netCDF4
except ImportError:
    pass

try:
    import xarray as xr
except ImportError:
    pass

__all__ = [
    'Array',
    'ArrayGroup',
]

unit_mapper = {
    "nanoseconds": "ns",
    "microseconds": "us",
    "milliseconds": "ms",
    "seconds": "s",
    "hours": "h",
    "minutes": "m",
    "days": "d",
}


class InvalidUnitString(Exception):
    def __init__(self, *args, **kwargs):
        super(InvalidUnitString, self).__init__(*args, **kwargs)


def num2date(times, units, calendar=None):
    """Convert an array of integers into datetime objects.

    This function optimizes the num2date function of python-netCDF4 if the
    standard calendar is used.

    Args:
        times: An array of integers representing timestamps.
        units: A string with the format "{unit} since {epoch}",
            e.g. "seconds since 1970-01-01T00:00:00".
        calendar: (optional) Standard is gregorian. If others are used,
            netCDF4.num2date will be called.

    Returns:
        Either an array of numpy.datetime64 objects (if standard gregorian
        calendar is used), otherwise an array of python datetime objects.
    """
    try:
        unit, epoch = units.split(" since ")
    except ValueError:
        raise InvalidUnitString("Could not convert to datetimes!")

    if calendar is None:
        calendar = "gregorian"
    else:
        calendar = calendar.lower()

    if calendar != "gregorian":
        return netCDF4.num2date(times, units, calendar).astype(
            "M8[%s]" % unit_mapper[unit])

    # Numpy uses the epoch 1970-01-01 natively.
    converted_data = times.astype("M8[%s]" % unit_mapper[unit])
    # Maybe there is another epoch used?
    if np.datetime64(epoch) != np.datetime64("1970-01-01"):
        converted_data -= np.datetime64("1970-01-01") - np.datetime64(epoch)
    return converted_data


def date2num(dates, units, calendar=None):
    """Convert an array of integer into datetime objects.

    This function optimizes the date2num function of python-netCDF4 if the
    standard calendar is used.

    Args:
        dates: Either an array of numpy.datetime64 objects (if standard
            gregorian calendar is used), otherwise an array of python
            datetime objects.
        units: A string with the format "{unit} since {epoch}",
            e.g. "seconds since 1970-01-01T00:00:00".
        calendar: (optional) Standard is gregorian. If others are used,
            netCDF4.num2date will be called.

    Returns:
        An array of integers.
    """
    if calendar is None:
        calendar = "gregorian"
    else:
        calendar = calendar.lower()

    if calendar != "gregorian":
        return netCDF4.date2num(dates, units, calendar)

    try:
        unit, epoch = units.split(" since ")
    except ValueError:
        raise InvalidUnitString("Could not convert to numeric values!")

    converted_data = \
        dates.astype("M8[%s]" % unit_mapper[unit]).astype("int")
    if np.datetime64(epoch) != np.datetime64("1970-01-01"):
        converted_data -= np.datetime64("1970-01-01") - np.datetime64(epoch)
    return converted_data


class Array(np.ndarray):
    """An extended numpy array with attributes and dimensions.

    """

    def __new__(cls, data, attrs=None, dims=None):
        obj = np.asarray(data).view(cls)

        if attrs is not None:
            obj.attrs = attrs

        if dims is not None:
            obj.dims = dims

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.attrs = getattr(obj, "attrs", {})

        # Update the number of dimensions if the shape has been changed.
        if not hasattr(obj, "dims") or len(obj.dims) != len(self.shape):
            self.dims = ["dim_%d" % i for i in range(len(self.shape))]
        else:
            self.dims = obj.dims

        # self.dims = getattr(
        #     obj, 'dims',
        #     [None for _ in range(len(self.shape))]
        # )

    # To make comparisons of datetime64 arrays with string or python
    # datetime objects possible, we complement the functionality of
    # comparison operators.
    def __lt__(self, other):
        return self._complement_comparisons_of_datetime64(
            super(Array, self).__lt__, other,
        )

    def __le__(self, other):
        return self._complement_comparisons_of_datetime64(
            super(Array, self).__le__, other,
        )

    def __eq__(self, other):
        return self._complement_comparisons_of_datetime64(
            super(Array, self).__eq__, other,
        )

    def __ge__(self, other):
        return self._complement_comparisons_of_datetime64(
            super(Array, self).__ge__, other,
        )

    def __gt__(self, other):
        return self._complement_comparisons_of_datetime64(
            super(Array, self).__gt__, other,
        )

    def __ne__(self, other):
        return self._complement_comparisons_of_datetime64(
            super(Array, self).__ne__, other,
        )

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        # TODO: Sometimes this crashes because the shape attribute has no items
        try:
            if self.shape[0] < 5:
                items = np.array_str(self[:self.shape[0]])
            else:
                items = ", ".join([
                    str(self[0]), str(self[1]), ".. ",
                    str(self[-2]), str(self[-1])])
            info = "[{}, dtype={}]".format(items, self.dtype)
            info += "\nDimensions: "
            info += ", ".join(
                ["%s (%d)" % (dim, self.shape[i])
                 for i, dim in enumerate(self.dims)])
        except IndexError:
            info = np.array_str(self)

        if self.attrs:
            info += "\nAttributes:"
            for attr, value in self.attrs.items():
                info += "\n\t{} : {}".format(attr, value)

        return info

    def _complement_comparisons_of_datetime64(self, method, other):
        """Complement the comparison of a datetime64 array with a time
        string or python datetime object.

        Args:
            method: __gt__ or __lt__, etc.
            other: Other object with that you want to compare.

        Returns:

        """

        if str(self.dtype).startswith("datetime64"):
            if isinstance(other, (str, datetime)):
                return method(np.datetime64(other))

        # Default comparison:
        return method(other)


    # def __repr__(self):
    #     return self.__str__()

    def apply_on_bins(self, bins, functions, return_dict=False):
        """A convenient function to apply functions on a binned array.

        Args:
            bins: List of lists which contain the indices for the bins.
            functions: Must be a dictionary of names (keys) and function
                references (values).
            return_dict: If true, a dictionary instead of an ArrayGroup will be
                returned.

        Returns:
            An ArrayGroup or dictionary with the return values.
        """
        binned_data = self.bin(bins)

        if return_dict:
            return_values = {
                name: np.asarray(
                    [func(bin, 0) for bin in binned_data]
                ).flatten()
                for name, func in functions.items()
            }
        else:
            return_values = ArrayGroup()
            for name, func in functions.items():
                return_values[name] = np.asarray([
                    func(bin, 0) for bin in binned_data]
                ).flatten()

        return return_values

    def average(self, window_size):
        """Calculates the sliding average for this array.

        Args:
            window_size: The size of the sliding window.

        Returns:
            Array with the averaged values.

        Examples:

        """

        # This code is taken from https://stackoverflow.com/a/15956341
        padded = np.pad(
            self, (0, window_size - self.size % window_size),
            mode='constant', constant_values=np.NaN
        )
        return np.nanmean(padded.reshape(-1, window_size), axis=1)

    def bin(self, bins):
        return [
            self[indices]
            for i, indices in enumerate(bins)
        ]

    @classmethod
    def from_xarray(cls, xarray_object):
        return cls(xarray_object.data, xarray_object.attrs, xarray_object.dims)

    def group(self):
        """Groups all elements and returns their appearances.

        This works with a pretty efficient algorithm posted in
        https://stackoverflow.com/a/23271510.

        Returns:
            A dictionary with the elements as keys and a list of their indices
            as values.

        Examples:
            .. :code-block:: python
            array = Array([0, 0, 1, 2, 2, 4, 2, 6])
            groups = array.group()
            print(groups)
            # Prints:
            # {0: Array([0, 1]), 1: Array([2]), 2: Array([3, 4, 6]),
            # 4: Array([5]), 6: Array([7])}
        """
        sort_idx = np.argsort(self)
        sorted_array = self[sort_idx]
        unq_first = np.concatenate(
            ([True], sorted_array[1:] != sorted_array[:-1]))
        unq_items = sorted_array[unq_first]
        unq_count = np.diff(np.nonzero(unq_first)[0])
        unq_idx = np.split(sort_idx, np.cumsum(unq_count))
        return dict(zip(unq_items, unq_idx))

    def remove_duplicates(self):
        return pd.unique(self)

    def to_string(self):
        return super(Array, self).__str__()

    def to_xarray(self):
        return xr.DataArray(self, attrs=self.attrs, dims=self.dims)


class ArrayGroup:
    """A specialised dictionary for arrays.

    Still under development.
    """

    def __init__(self, name=None, hidden_prefix=None):
        """Initializes an ArrayGroup object.

        Args:
            name: Name of the ArrayGroup as string.
            hidden_prefix: (optional) Define the prefix of hidden groups or
                variables. The default is "__".
        """

        self.attrs = {}

        if hidden_prefix is None:
            self.hidden_prefix = "__"
        else:
            self.hidden_prefix = hidden_prefix

        # All variables (excluding groups) will be saved into this dictionary:
        self._vars = {}

        # All groups will be saved here.
        self._groups = {}

        if name is None:
            self.name = "{}, {}:".format(type(self), id(self))
        else:
            self.name = name

    def __contains__(self, item):
        var, rest = self.parse(item)
        if var == "/":
            return True

        if var in self._vars:
            return True

        if var in self._groups:
            if rest:
                return rest in self[var]
            return True

        return False

    def __iter__(self):
        self._iter_vars = self.vars(deep=True)
        return self

    def __next__(self):
        return next(self._iter_vars)

    def __delitem__(self, key):
        var, rest = self.parse(key)

        # If the user tries to delete all variables:
        if not var:
            raise KeyError("The main group cannot be deleted. Use the clear "
                           "method to delete all variables and groups.")

        if rest:
            del self._groups[var][rest]
            return

        if var in self._vars:
            del self._vars[var]
        else:
            try:
                del self._groups[var]
            except KeyError:
                raise KeyError(
                    "Cannot delete! There is neither a variable nor group "
                    "named '{}'!".format(var))

    def __getitem__(self, item):
        """Enables dictionary-like access to the ArrayGroup.

        There are different ways to access one element of this ArrayGroup:

        * by *array_group["var"]*: returns a variable (Array) or group
            (ArrayGroup) object.
        * by *array_group["group1/var"]*: returns the variable *var* from the
            group *group1*.
        * by *array_group["/"]*: returns this object itself.
        * by *array_group[0:10]*: returns a copy of the first ten elements
            for each variable in the ArrayGroup object. Note: all variables
            should have the same length.
        * by *array_group[("var1", "var2", )]*: selects the fields "var1"
            and "var2" from the array group and returns them as a new array
            group object.

        Args:
            item: Can be a string, integer, slice or tuple of strings.

        Returns:
            Either an Array or an ArrayGroup object.
        """

        # Accessing via key:
        if isinstance(item, str):
            var, rest = self.parse(item)

            # All variables are requested (return the object itself)
            if not var:
                return self

            if not rest:
                if var in self._vars:
                    return self._vars[var]

                try:
                    return self._groups[var]
                except KeyError:
                    raise KeyError(
                        "There is neither a variable nor group named "
                        "'{}'!".format(var))
            else:
                if var in self._groups:
                    return self._groups[var][rest]
                else:
                    raise KeyError("'{}' is not a group!".format(var))
        else:
            # Selecting elements via slicing:
            return self.select(item)

    def __bool__(self):
        for var, data in self.items(deep=True):
            if data.size:
                return True

        return False

    def __setitem__(self, key, value):
        var, rest = self.parse(key)

        if not var:
            raise ValueError("You cannot change the main group directly!")

        if not rest:
            # Try automatic conversion from numpy array to Array.
            if not isinstance(value, (Array, ArrayGroup, type(self))):
                value = Array(value)

            if isinstance(value, Array):
                self._vars[var] = value

                # Maybe someone wants to create a variable with a name that
                # has been the name of a group earlier?
                if var in self._groups:
                    del self._groups[var]
            else:
                self._groups[var] = value

                # May be someone wants to create a group with a name that
                # has been the name of a variable earlier?
                if var in self._vars:
                    del self._vars[var]
        else:
            # Auto creation of groups
            if var not in self._groups:
                self._groups[var] = type(self)(name=var)

                # May be someone wants to create a group with a name that
                # has been the name of a variable earlier?
                if var in self._vars:
                    del self._vars[var]
            self._groups[var][rest] = value

    def __str__(self):
        info = "Name: {}\n".format(self.name)
        info += "  Attributes:\n"
        if self.attrs:
            for attr, value in self.attrs.items():
                info += "    {} : {}\n".format(attr, value)
        else:
            info += "    --\n"

        info += "  Groups:\n"
        if self._groups:
            for group in self.groups(deep=True):
                info += "    {}\n".format(group)
        else:
            info += "    --\n"

        info += "  Variables:\n"
        variables = list(self.vars(deep=True))
        if variables:
            coords = self.coords(deep=True)
            for var in variables:
                info += "    {}{}:\n{}\n".format(
                    var, " (coord)" if var in coords else "",
                    textwrap.indent(str(self[var]), ' ' * 6)
                )
        else:
            info += "  --\n"

        return info

    def apply(self, func_with_args, deep=False, new_object=False):
        """Apply a function to all variables.

        Args:
            func_with_args: Tuple of reference to the function and arguments.
            deep: Apply the function also on variables of subgroups.
            new_object: If this is true, a new ArrayGroup will be created
                with variables of the return values. Otherwise the return
                value is simply a dictionary with the variables names and the
                return values.

        Returns:
            An ArrayGroup or dictionary object with the return values.
        """
        if new_object:
            new_data = type(self)()
            for var, data in self.items(deep):
                new_data[var] = func_with_args[0](data, **func_with_args[1:])
        else:
            new_data = {
                var: func_with_args[0](data, **func_with_args[1:])
                for var, data in self.items(deep)
            }

        return new_data

    def as_type(self, new_type):
        new_object = new_type()
        new_object.__dict__ = self.__dict__
        return new_object

    def collapse(self, bins, collapser=None, deep=False):
        """Divide the data of each variables in bins and apply a function to
        them.

        Args:
            bins: List of lists which contain the indices for the bins.
            collapser: Function that should be applied on each bin (
                numpy.nanmean is the default).
            deep: Collapses also the variables of the subgroups.

        Returns:
            One ArrayGroup object with the collapsed data.
        """
        # Default collapser is the mean function:
        if collapser is None:
            collapser = np.nanmean

        bins = np.asarray(bins)

        # Collapse the data:
        collapsed_data = type(self)()
        collapsed_data.attrs.update(**self.attrs)
        for var, data in self.items(deep):
            # The data could contain datetime objects. A numerical collapser
            # function will crash with such an object. Hence, we convert the
            # datetime objects to floats temporarily.
            if isinstance(data.item(0), datetime):
                numerical_data = Array([d.timestamp() for d in data])
                binned_data = numerical_data.bin(bins)
                collapsed_data[var] = \
                    [datetime.fromtimestamp(collapser(bin, 0))
                     for bin in binned_data]
            else:
                binned_data = data.bin(bins)
                collapsed_data[var] = \
                    [collapser(bin, 0) for bin in binned_data]

            collapsed_data[var].attrs.update(**data.attrs)
        return collapsed_data

    @classmethod
    def concatenate(cls, objects, dimension=None):
        """Concatenate multiple ArrayGroup objects.

        Notes:
            The attribute and dimension information of some arrays may get
            lost.

        Args:
            objects: List of GeoData objects to concatenate.
            dimension: Dimension on which to concatenate.

        Returns:
            A
        """
        new_data = cls()
        for var in objects[0]:
            if isinstance(objects[0][var], cls):
                new_data[var] = cls.concatenate(
                    [obj[var] for obj in objects],
                    dimension)
            else:
                if dimension is None:
                    dimension = 0
                new_data[var] = np.concatenate(
                    [obj[var] for obj in objects],
                    dimension)

        return new_data

    def coords(self, deep=False):
        """Returns all variable names that are used as dimensions for other
        variables.

        Args:
            deep:

        Returns:

        """
        variables = list(self.vars(deep))
        coords = [
            coord
            for coord in variables
            for var in variables
            if coord in self[var].dims
        ]
        return list(set(coords))

    def drop(self, fields, inplace=True):
        """Remove fields from the object.

        Args:
            fields:
            inplace:

        Returns:
            An ArrayGroup without the dropped fields.
        """
        if inplace:
            obj = self
        else:
            obj = copy.deepcopy(self)

        for field in list(fields):
            del obj[field]

        return obj

    @classmethod
    def from_dict(cls, dictionary):
        """Create an ArrayGroup from a dictionary.

        Args:
            dictionary: Dictionary-like object of Arrays or numpy.arrays.

        Returns:
            An ArrayGroup object.
        """
        obj = cls()
        for var, data in dictionary.items():
            obj[var] = data

        return obj

    @classmethod
    def from_netcdf(cls, filename, fields=None, convert_times=True,):
        """Creates an ArrayGroup object from a netCDF file.

        Args:
            filename: Path and file name from where to load a new ArrayGroup.
                Can also be a tuple/list of file names.
            fields: (optional) List or tuple of variable or
                group names). Only those fields are going to be read.
            convert_times: Set this to true if you want to convert time
                fields into datetime objects.

        Returns:
            An ArrayGroup object.
        """

        if isinstance(filename, (tuple, list)):
            opener = netCDF4.MFDataset
        else:
            opener = netCDF4.Dataset

        with opener(filename, "r") as root:
            return cls._get_group_from_netcdf_group(
                root, fields, convert_times
            )

    @classmethod
    def _get_group_from_netcdf_group(cls, group, fields, convert_times):
        array_group = cls()

        array_group.attrs.update(**group.__dict__)

        # Limit the reading of the file to some fields:
        if fields is not None:
            for field in fields:
                if isinstance(group[field], netCDF4._netCDF4.Group):
                    array_group[field] = \
                        cls._get_group_from_netcdf_group(
                            field, None, convert_times)
                else:
                    array_group[field] = \
                        ArrayGroup._get_variable_from_netcdf_group(
                            group[field], convert_times
                        )
            return array_group

        # Otherwise, we want to read all variables and groups:
        # Add the variables:
        for var, data in group.variables.items():
            array_group[var] = ArrayGroup._get_variable_from_netcdf_group(
                data, convert_times
            )

        # Add the groups
        for subgroup, subgroup_obj in group.groups.items():
            array_group[subgroup] = \
                cls._get_group_from_netcdf_group(
                    subgroup_obj, None, convert_times)

        return array_group

    @staticmethod
    def _get_variable_from_netcdf_group(nc_var, convert_times):
        # There might be empty fields:
        if not nc_var:
            return Array(
                [], attrs=nc_var.__dict__,
                dims=nc_var.dimensions,
            )

        # Handle time fields differently
        if convert_times and "units" in nc_var.__dict__:
            try:
                data = num2date(nc_var[:], nc_var.units)
            except InvalidUnitString:
                # This means it is no time variable
                data = nc_var[:]
            except Exception as e:
                raise e
        else:
            data = nc_var[:]

        return Array(
            data, attrs=nc_var.__dict__,
            dims=nc_var.dimensions,
        )

    @classmethod
    def from_xarray(cls, xarray_object):
        """Creates an ArrayGroup object from a xarray.Dataset object.

        Args:
            xarray_object: A xarray.Dataset object.

        Returns:
            An ArrayGroup object.
        """

        array_dict = cls()
        for var in xarray_object:
            array_dict[var] = Array.from_xarray(xarray_object[var])

        array_dict.attrs.update(**xarray_object.attrs)

        return array_dict

    def get_coords(self, field):
        """Gets the coordinates for a field.

        Args:
            field: Name of the field.

        Returns:

        """
        return_list = []

        coords = self.coords()
        for i, coord in enumerate(self[field].dims):
            if coord in coords:
                return_list.append(self[coord])
            else:
                return_list.append(np.arange(len(self[field].shape[i])))

        return return_list

    def get_range(self, field, deep=False, axis=None):
        """Get the minimum and maximum of one field.

        Args:
            field: Name of the variable.
            deep: Including also the fields in subgroups (not only main group).
            axis: Axis where to calculate the minimum or maximum from.

        Returns:
            The minimum and the maximum value of a field.

        Examples:

        .. code-block:: python

            # Imagine you have an ArrayGroup with multiple subgroups, each with
            # a field "time". You want to have the lowest and the highest time
            # value of all subgroups. Simply do:
            ag = ArrayGroup()
            ag["group1/time"] = np.arange(10)
            ag["group2/time"] = np.arange(100, 200)
            ag["group3/time"] = np.arange(50, 250)
            print(ag.get_range("time", deep=True))
            # Prints: (0, 250)
        """
        variables = list(self.vars(deep, with_name=field))

        if not variables:
            raise KeyError("No variable named '%s' was found!")

        start = [np.nanmin(self[var], axis).item(0) for var in variables]
        end = [np.nanmax(self[var], axis).item(0) for var in variables]
        return min(start), max(end)

    def groups(self, deep=False, exclude_prefix=None):
        """Returns the names of all groups in this GeoData object.

        Args:
            deep: Including also subgroups (not only main group).
            exclude_prefix: All group names starting with this prefix are not
                going to be returned.

        Yields:
            Name of group.
        """

        for group in self._groups:
            if exclude_prefix is None or not group.startswith(exclude_prefix):
                yield group
            if deep:
                yield from (group + "/" + subgroup
                            for subgroup in self[group].groups(deep))

    def is_group(self, name):
        return name in self._groups

    def is_var(self, name):
        return name in self._vars

    def items(self, deep=False):
        """Iterate over all pairs of variables and their content.

        Args:
            deep: Including also variables from the subgroups.

        Yields:
            Tuple of variable name and content.
        """
        for var in self.vars(deep):
            yield var, self[var]

    @staticmethod
    def _level(var):
        level = len(var.split("/")) - 1
        if var.startswith("/"):
            level -= 1
        return level

    def limit_by(self, field, lower_bound=None, upper_bound=None):
        """Extract the parts of this Arraygroup where *field* lies between two
        bounds.

        This works only if all first dimensions of each variable have the same
        length.

        Args:
            field: A name of a variable.
            lower_bound: A number / object as lower bound.
            upper_bound: A number / object as upper bound.

        Returns:
            New limited ArrayGroup object.

        Examples:

            .. code-block:: python

            ag = ArrayGroup()
        """
        if lower_bound is not None and upper_bound is not None:
            indices = (self[field] >= lower_bound) \
                      & (self[field] <= upper_bound)
        elif lower_bound is None:
            indices = self[field] <= upper_bound
        elif upper_bound is None:
            indices = self[field] >= lower_bound
        else:
            raise ValueError("One bound must be set!")
        return self[indices]

    @classmethod
    def merge(cls, objects, groups=None, overwrite_error=True):
        """Merges multiple ArrayGroup objects to one.

        Notes:
            Merging of sub groups with the same name does not work properly.

        Args:
            objects: List of GeoData objects.
            groups: List of strings. You can give each object in
                :param:`objects` a group. Must have the same length as
                :param:`objects`.
            overwrite_error: Throws a KeyError when trying to merge`
                ArrayGroups containing same keys.

        Returns:
            An ArrayGroup object.
        """
        inserted = set()
        merged_data = cls()
        for i, obj in enumerate(objects):
            merged_data.attrs.update(**obj.attrs)
            for var in obj.vars(deep=True):
                if overwrite_error and var in inserted:
                    raise KeyError("The variable '{}' occurred multiple "
                                   "times!".format(var))
                else:
                    if groups is not None:
                        if groups[i] not in merged_data:
                            merged_data[groups[i]] = cls()
                        merged_data[groups[i]][var] = obj[var]
                    else:
                        merged_data[var] = obj[var]

        return merged_data

    @staticmethod
    def parse(path, root=True):
        """Parses *path* into first group and rest.

        You can access the groups and fields via different keys:

        * "value": Returns ("value", "")
        * "/value": Returns ("value", "")
        * "value1/value2/value3": Returns ("value1", "value2/value3")
        * "value/": Returns ("value", "")
        * "/": Returns ("", "")

        Args:
            path:
            root: If true, it splits the path between the root group and the
                rest. If false, it returns the full group name and the
                variable name.

        Returns:

        """
        if path.startswith("/"):
            return path[1:], ""

        if "/" not in path:
            return path, ""

        if root:
            return path.split("/", 1)
        else:
            return path.rsplit("/", 1)

    def plot(self, fields=None, ptype=None, fig=None, layout=None,
             **plotting_args):
        """

        Args:
            fields: List of variables that you want to plot.
            ptype: Plot type. So far these are possible: *plot* (default),
                *scatter* or *histogram*.
            fig:
            layout: (optiona) The positioning of the plots. Must be a tuple
                of two integers (rows, columns).
            **plotting_args: Additional plotting keyword arguments for
                matplotlib routine.

        Returns:

        """
        ptypes = {
            # name : [func, number of variables]
            "histogram": plt.hist,
            "plot": plt.plot,
            "scatter": plt.scatter,
        }

        if ptype not in ptypes:
            raise ValueError("Unknown plot type '%s'" % ptype)

        if fields is None:
            fields = list(self.vars())

        if layout is None:
            if len(fields) % 4 == 0 and len(fields) > 4:
                rows = int(len(fields) / 4)
                cols = 4
            elif len(fields) % 2 == 0:
                rows = int(len(fields) / 2)
                cols = 2
            elif len(fields) % 3 == 0:
                rows = int(len(fields) / 3)
                cols = 3
            else:
                rows = int(len(fields) / 3) + 1
                cols = 3
        else:
            rows, cols = layout

        if fig is None:
            fig, axes = plt.subplots(rows, cols)
        else:
            axes = fig.subplots(rows, cols)

        axes = list(chain.from_iterable(axes))

        if ptype == "histogram":
            if "bins" not in plotting_args:
                plotting_args["bins"] = 50

        for i, field in enumerate(fields):
            data = self[field]
            if len(data.dims) != 1:
                warnings.warn("Cannot plot multi-dimensional data!")
                continue

            axes[i].set_title(field)

            if ptype == "plot":
                coord = self.get_coords(field)[0]
                axes[i].plot(coord, data)
            elif ptype == "scatter":
                coord = self.get_coords(field)[0]
                axes[i].scatter(coord, data)
            elif ptype == "histogram":
                axes[i].hist(data[~np.isnan(data)], **plotting_args)

            axes[i].grid()

        fig.tight_layout()

        return axes

    def rename(self, mapping, inplace=True):
        if inplace:
            obj = self
        else:
            obj = copy.deepcopy(self)

        for old_name, new_name in mapping.items():
            array = obj[old_name]
            del obj[old_name]
            obj[new_name] = array

        return obj

    def select(self, indices_or_fields, inplace=False):
        """Select an TODO.

        Args:
            indices_or_fields:
            inplace:

        Returns:

        """
        if isinstance(indices_or_fields, str):
            raise TypeError("For field selection indices_or_fields must be "
                            "a tuple/list of strings.")

        if inplace:
            obj = self
        else:
            obj = type(self)()
            obj.attrs.update(**self.attrs)

        # Try selecting by indices or slices:
        try:
            for var in self.vars(True):
                obj[var] = self[var][indices_or_fields]
        except IndexError as e:
            fields = list(indices_or_fields)
            if isinstance(fields[0], str):
                if inplace:
                    # We want to keep the original object and simply drop all
                    # unwanted variables.
                    unwanted_vars = set(obj.vars(True)) - set(fields)
                    obj.drop(unwanted_vars, inplace=True)
                else:
                    for var in self.vars(True):
                        if var in indices_or_fields:
                            obj[var] = self[var]
            else:
                raise IndexError(
                    str(e) + "\nCould not select parts of '%s'.\n" % var)

        return obj

    def to_dict(self, deep=True):
        """Exports variables to a dictionary.

        Args:
            deep: Export also variables from the subgroups.

        Returns:
            A dictionary object.
        """
        return {var: data for var, data in self.items(deep)}

    def to_netcdf(self, filename, attribute_warning=True,
                  avoid_dimension_errors=True):
        """Stores the ArrayGroup to a netcdf4 file.

        Args:
            filename: Path and file name to which to save this object.
            attribute_warning: Attributes in netCDF4 files may only be a
                number, list or string. If this is true, this method gives a
                warning whenever it tries to store an attribute not fulfilling
                these conditions.
            avoid_dimension_errors: This method raises an error if two
                variables use the same dimension but expecting different
                lengths. If this parameter is true, the error will not be
                raised but an additional dimension will be created.

        Returns:
            None
        """

        with netCDF4.Dataset(filename, "w", format="NETCDF4") as root_group:
            # Add all variables of the main group:
            self._add_group_to_netcdf(
                "/", root_group, attribute_warning, avoid_dimension_errors)

            # Add all variables of the sub groups:
            for group in self.groups(deep=True):
                nc_group = root_group.createGroup(group)
                self._add_group_to_netcdf(
                    group, nc_group, attribute_warning,
                    avoid_dimension_errors)

    def _add_group_to_netcdf(
            self, group, nc_group, attr_warning, avoid_dimension_errors):
        for attr, value in self[group].attrs.items():
            try:
                setattr(nc_group, attr, value)
            except TypeError:
                if attr_warning:
                    warnings.warn(
                        "Cannot store attribute '{}' since it is not "
                        "a number, list or string!".format(attr))

        coords = self[group].coords()
        for var, data in self[group].items():
            # Coordinates should be saved in the end, otherwise a netCDF error
            # will be raised.
            if var in coords:
                continue

            self._add_variable_to_netcdf_group(
                var, data, nc_group, attr_warning, avoid_dimension_errors
            )

        for coord in coords:
            data = self[group][coord]

            self._add_variable_to_netcdf_group(
                coord, data, nc_group, attr_warning, avoid_dimension_errors
            )

    @staticmethod
    def _add_variable_to_netcdf_group(
            var, data, nc_group, attr_warning, avoid_dimension_errors):
        for i, dim in enumerate(data.dims):
            if dim not in nc_group.dimensions:
                nc_group.createDimension(
                    dim, data.shape[i]
                )
            elif data.shape[i] != len(nc_group.dimensions[dim]):
                # The dimension already exists but have a different
                # length than expected. Either we raise an error or we
                # create a new dimension for this variable.
                if not avoid_dimension_errors:
                    raise ValueError(
                        "The dimension '{}' already exists and does not "
                        "have the same length as the same named dimension "
                        "from the variable '{}'. Maybe you should consider"
                        " renaming it?".format(dim, var))
                else:
                    while dim in nc_group.dimensions:
                        dim += "0"
                    nc_group.createDimension(
                        dim, data.shape[i]
                    )
                    data.dims[i] = dim

        # Try to catch up time objects:
        if str(data.dtype).startswith("datetime64"):
            nc_var = nc_group.createVariable(
                var, "f8", data.dims
            )
            nc_var.units = \
                "seconds since 1970-01-01T00:00:00Z"
            time_data = date2num(data, nc_var.units)
            nc_var[:] = time_data
        elif isinstance(data.item(0), datetime):
            nc_var = nc_group.createVariable(
                var, "f8", data.dims
            )
            # TODO: Per default we save seconds since blabla. Maybe this should
            # TODO: be dynamic?
            nc_var.units = \
                "seconds since 1970-01-01T00:00:00Z"
            time_data = netCDF4.date2num(data, nc_var.units)
            nc_var[:] = time_data
        else:
            if str(data.dtype) == "bool":
                data = data.astype("int")
            try:
                nc_var = nc_group.createVariable(
                    var, data.dtype, data.dims
                )
                nc_var[:] = data
            except TypeError as e:
                raise TypeError("Tried to save '{}': {}".format(var, str(e)))

        for attr, value in data.attrs.items():
            # Do not overwrite already set attributes
            if hasattr(nc_var, attr):
                continue
            try:
                setattr(nc_var, attr, value)
            except TypeError:
                if attr_warning:
                    warnings.warn(
                        "Cannot store attribute '{}' since it is not "
                        "a number, list or string!".format(attr))

    def to_xarray(self):
        """Converts this ArrayGroup object to a xarray.Dataset.

        Warnings:
            Still contain bugs!

        Returns:
            A xarray.Dataset object.
        """

        xarray_object = xr.Dataset()
        for var, data in self.items(deep=True):
            xarray_object[var] = data.to_xarray()

        xarray_object.attrs.update(**self.attrs)

        return xarray_object

    def values(self, deep=False):
        for var in self.vars(deep):
            yield self[var]

    def vars(self, deep=False, with_name=None, hidden=True):
        """Returns the names of all variables in this GeoData object main
        group.

        Args:
            deep: Searching also in subgroups (not only main group).
            with_name: (optional) Only the variables with this base name
                will be returned (makes only sense when *deep* is true).
            hidden: (optional) If false, all variables which start with the
                defined hidden prefix will be not returned. If *with_name* is
                given, this will be ignored.

        Yields:
            Full name of one variable (including group name).
        """

        # Only the variables of the main group:
        if with_name is None:
            if hidden or self.hidden_prefix is None:
                yield from self._vars
            else:
                yield from filter(
                    lambda x: not x.startswith(self.hidden_prefix), self._vars)
        elif with_name in self._vars:
            yield with_name

        if deep:
            for group in self._groups:
                yield from (
                    group + "/" + sub_var
                    for sub_var in self[group].vars(
                        deep, with_name, hidden)
                )
