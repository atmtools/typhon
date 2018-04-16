"""Contains classes that extend the functionality of plain numpy ndarrays,
to bundle them in labeled groups and store them to netCDF4 files.
"""

import copy
import glob
import textwrap

import netCDF4
import xarray as xr

__all__ = [
    'DataGroup',
]


class DataGroup:

    def __init__(self, data=None, name=None):
        if data is None:
            self._data = xr.Dataset()
        else:
            self._data = data
        self._groups = {}

        self.name = "DataGroup" if name is None else name

    def __bool__(self):
        if self.data:
            return True

        return bool(self.groups)

    def __contains__(self, item):
        if item in self._data:
            return True

        return item in self._groups

    def __delitem__(self, path):
        # The path may contain the group and a variable name
        group, rest = DataGroup.parse(path)

        # If the user tries to delete all variables:
        if not group:
            raise KeyError("The main group cannot be deleted. Use the clear "
                           "method to delete all variables and groups.")

        if rest:
            del self._groups[group][rest]
            return

        if group in self._data:
            del self._data[group]
        else:
            try:
                del self._groups[group]
            except KeyError:
                raise KeyError(
                    f"Cannot delete! There is neither a variable nor group "
                    "named '{group}'!")

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, path):
        if path == slice(None, None, None):
            return self._data

        if isinstance(path, (tuple, list, set)):
            return self.select(path)

        # The path may contain the group and a variable name
        group, rest = DataGroup.parse(path)

        if not group:
            return self

        if rest:
            # `group` must be a valid group name
            return self._groups[group][rest]

        # `group`could be a group or a variable name. We check for
        # variables first:
        var = self._data.get(group, None)
        if var is not None:
            return var

        # Okay, now `group`must be a group name:
        return self._groups[group]

    def __setitem__(self, path, value):
        if path == slice(None, None, None):
            if isinstance(value, xr.Dataset):
                self.data = value
            else:
                raise ValueError("Root group must be a xarray.Dataset!")
            return

        # The path may contain the group and a variable name
        group, rest = DataGroup.parse(path)

        if not group:
            raise ValueError("Root group cannot be changed via '/'. Use ':' "
                             "instead!")

        if rest:
            # If no group with this name exists, we simply create a new one:
            try:
                subgroup = self._groups[group]
            except KeyError:
                self._groups[group] = DataGroup()
                subgroup = self._groups[group]

            subgroup[rest] = value
        else:
            # `group` may be meant as group or variable name:
            if isinstance(value, DataGroup):
                self._groups[group] = value

                # May be someone wants to create a group with a name that
                # has been the name of a variable earlier?
                if group in self._data:
                    del self._data[group]
            elif isinstance(value, xr.Dataset):
                self._groups[group] = DataGroup(data=value)

                # May be someone wants to create a group with a name that
                # has been the name of a variable earlier?
                if group in self._data:
                    del self._data[group]
            else:
                self._data[group] = value

                # May be someone wants to create a variable with a name that
                # has been the name of a group earlier?
                if group in self._groups:
                    del self._groups[group]

    def __repr__(self):
        return str(self)

    def __str__(self):
        info = f"Name: {self.name}\n"
        info += textwrap.indent(repr(self.data), ' ' * 2)
        info += "\n"
        for name, group in self.groups.items():
            info += textwrap.indent(f"{name}/\n", ' ' * 2)
            info += textwrap.indent(repr(group), ' ' * 4)

        return info

    def apply(self, method, *args, **kwargs):
        dg = type(self)()
        dg[:] = method(self[:], *args, **kwargs)
        for group in self.deep("groups"):
            dg[group] = method(self[group].data, *args, **kwargs)

        return dg

    @classmethod
    def concat(cls, objs, **kwargs):
        dg = cls()

        if not objs:
            return dg
        if len(objs) == 1:
            return objs[0].copy()

        # Concatenate the main groups:
        dg[:] = xr.concat([obj[:] for obj in objs], **kwargs)
        # Concatenate the other groups:
        for group in objs[0].deep("groups"):
            dg[group] = xr.concat([obj[group].data for obj in objs], **kwargs)

        return dg

    def copy(self):
        return copy.deepcopy(self)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def deep(self, ftype=None, name=None):
        if ftype is None or ftype == "vars":
            for var in self:
                if name is None or name == var:
                    yield var
            for group in self.deep("groups"):
                for var in self[group]:
                    if name is None or name == var:
                        yield group + "/" + var
        elif ftype == "groups":
            for group in self.groups:
                yield group
                for subgroup in self[group].deep("groups"):
                    if name is None or name == subgroup:
                        yield group + "/" + subgroup
        else:
            raise ValueError(f"Unknown field type '{ftype}'!")

    def drop(self, fields, inplace=True):
        """Remove fields from the object

        Args:
            fields: List of field names (variables or groups)
            inplace:

        Returns:
            An DataGroup object without the dropped fields.
        """
        if inplace:
            obj = self
        else:
            obj = copy.deepcopy(self)

        for field in list(fields):
            del obj[field]

        return obj

    @classmethod
    def from_csv(cls, filename, fields=None, **csv_args):
        """Load an GroupedArrays object from a CSV file.

        Args:
            filename: Path and name of the file.
            fields: Fields to extract.
            **csv_args: Additional keyword arguments for the pandas function
                `pandas.read_csv`. See for more details:
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

        Returns:
            An GroupedArrays object.
        """
        data = pd.read_csv(filename, **csv_args).to_xarray()

        if fields is not None:
            data = data[fields]

        return cls(data)

    @classmethod
    def from_dict(cls, dictionary):
        dg = cls()
        for var, data in dictionary.items():
            dg[var] = data
        return dg

    @classmethod
    def from_netcdf(cls, paths, groups=None, **kwargs):
        """

        Args:
            paths:
            groups:
            **kwargs:

        Returns:

        """
        if "group" in kwargs:
            raise ValueError("Use `groups` instead of `group` as parameter!")

        dg = cls()

        if groups is None:
            # Find all groups by ourselves:
            if isinstance(paths, (tuple, list)):
                filename = paths[0]
            elif "*" in paths:
                filename = next(glob.iglob(paths))
            else:
                filename = paths

            with netCDF4.Dataset(filename, "r") as root:
                groups = list(DataGroup._netcdf_all_groups(root))

        for group in groups:
            if isinstance(paths, (tuple, list)) or "*" in paths:
                dg[group] = xr.open_mfdataset(paths, group=group, **kwargs)
            else:
                dg[group] = xr.open_dataset(paths, group=group, **kwargs)

        return dg

    @staticmethod
    def _netcdf_all_groups(top):
        for value in top.groups.values():
            yield value.name
            for children in DataGroup._netcdf_all_groups(value):
                yield value.name + "/" + children

    @property
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, value):
        self._groups = value

    def isel(self, *args, **kwargs):
        return self.apply(xr.Dataset.isel, *args, **kwargs)

    @staticmethod
    def parse(path, root=True):
        """Parse `path` into first group and rest.

        The first group may also a variable name.

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

    def rename(self, mapping, inplace=True, deep=False):
        if inplace:
            obj = self
        else:
            obj = copy.deepcopy(self)

        for old_name, new_name in mapping.items():
            array = obj[old_name]
            del obj[old_name]
            obj[new_name] = array

        if deep:
            for subgroup in obj.deep("groups"):
                obj[subgroup].rename(mapping, deep=True)

        return obj

    def sel(self, *args, **kwargs):
        return self.apply(xr.Dataset.sel, *args, **kwargs)

    def select(self, fields, inplace=False):
        if inplace:
            obj = self
        else:
            obj = type(self)()

        if inplace:
            # We want to keep the original object and simply drop all
            # unwanted variables.
            unwanted_vars = set(obj.deep()) - set(fields)
            self.drop(unwanted_vars, inplace=True)

            return self

        obj = type(self)()
        for field in fields:
            obj[field] = self[field]

        return obj
