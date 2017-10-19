# -*- coding: utf-8 -*-

"""General functions and classes for manipulating geographical data.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import typhon.plots

try:
    import xarray as xr
except:
    pass


__all__ = [
    'GeoData',
]


class GeoData:
    """Still under development. TODO.

    """

    def __init__(self, name=None):
        """A specialised container for geographical indexed data (with
        longitude, latitude and time field).

        Since the xarray.Dataset class cannot handle groups and is difficult to
        extend (see http://xarray.pydata.org/en/stable/internals.html#extending-xarray
        for more details).
        """

        self.attrs = {}

        self._groups = {}
        self._vars = {}

        if name is None:
            self.name = "<GeoData="+str(id(self))+">"
        else:
            self.name = name

    def __contains__(self, item):
        group, field = self.parse(item)
        if group not in self._fields:
            return False

        return field in self._fields[group]

    def __iter__(self):
        self._iter_vars = self.vars(deep=True)
        return self

    def __next__(self):
        return next(self._iter_vars)

    def __len__(self):
        if "time" not in self:
            return 0

        return len(self["time"])

    def __str__(self):
        info = self.name + "\n"
        for group in self.groups(deep=True):
            if group == "/":
                info += "- Main group:\n"
                info += "-" * len("- Main group:")
            else:
                info += "\n- " + group + ":\n"
                info += "-" * (len(group) + 3)

            info += "\n"

            for var in self.group(group):
                info += "\t{}: {}\n".format(var, self.group(group)[var])

        return info

    def __delitem__(self, key):
        group, field = self.parse(key)
        if not group:
        if not field:
            del self._groups[group]
        else:
            del self._fields[group][field]

    def __getitem__(self, item):
        group, var = self.parse(item)

        # The main group is requested:
        if not group and not var:
            return self._vars

        if not group:
            if var in self._vars:
                return self._vars[var]

            try:
                return self._groups[var]
            except KeyError:
                raise KeyError("There is neither a variable nor group named "
                               "'{}'!".format(var))

        if not var:
            return self._groups[group][var]

        return self._groups[group][var]

    def __setitem__(self, key, value):
        group, var = self.parse(key)

        if not group and not var:
            raise ValueError("You cannot change the main group directly!")

        if not group:
            if isinstance(value, xr.DataArray):
                self._vars[var] = value
            elif isinstance(value, GeoData):
                self._groups[group] = value
            else:
                raise ValueError("Value must be either a GeoData (group) "
                                 "or xarray.DataArray (variable) object!")

        if not var:
            if isinstance(value, xr.DataArray):
                self._vars[var] = value
            else:
                raise ValueError("Main group value must be a "
                                 "xarray.DataArray!")
            self._vars[group] = value
        else:

        self._fields[group][field] = value

    def bin(self, bins, fields=None, deep=False):
        new_data = GeoData()
        for var, data in self.items(deep):
            if fields is not None and var not in fields:
                continue
            binned_data = np.asarray([
                    data.data[indices]
                    for i, indices in enumerate(bins)
                ])
            data_dict = data.to_dict()
            data_dict["data"] = binned_data
            data_dict["dims"] = ["bin"] + list(data_dict["dims"])
            new_data[var] = xr.DataArray.from_dict(data_dict)
        return new_data

    def collapse(self, bins, collapser=None,
                 variation_filter=None, deep=False):
        # Default collapser is the mean function:
        if collapser:
            collapser = np.nanmean

        # Exclude all bins where the inhomogeneity (variation) is too high
        passed = np.ones_like(bins).astype("bool")
        if isinstance(variation_filter, tuple):
            if len(variation_filter) >= 2:
                if len(self[variation_filter[0]].dims) > 1:
                    raise ValueError("The variation filter can only be used "
                                     "for 1-dimensional data! I.e. the field "
                                     "'{}' must be 1-dimensional!".format(
                                        variation_filter[0]))

                # Bin only one field for testing of inhomogeneities:
                binned = self.bin(bins, fields=(variation_filter[0]))

                # The user can define a different variation function (
                # default is the standard deviation).
                if len(variation_filter) == 2:
                    variation = np.nanstd(binned[variation_filter[0]], 1)
                else:
                    variation = variation_filter[2](
                        binned[variation_filter[0]], 1)
                passed = variation < variation_filter[1]
            else:
                raise ValueError("The inhomogeneity filter must be a tuple "
                                 "of a field name, a threshold and a "
                                 "variation function (latter is optional).")

        bins = np.asarray(bins)

        # Before collapsing the data, we must bin it:
        collapsed_data = self.bin(bins[passed], deep)
        for var, data in collapsed_data.items(deep):
            data_dict = data.to_dict()
            data_dict["data"] = collapser(data_dict["data"], 1)
            data_dict["dims"] = data_dict["dims"][1:]
            collapsed_data[var] = xr.DataArray.from_dict(data_dict)
        return collapsed_data

    @staticmethod
    def concat(objects, dimension=None):
        new_data = GeoData()
        for var in objects[0]:
            if isinstance(objects[0][var], GeoData):
                new_data[var] = GeoData.concat([obj[var] for obj in objects])
            elif isinstance(objects[0][var], xr.DataArray):
                if dimension is None:
                    dimension = objects[0][var].dims[0]
                new_data[var] = xr.concat([obj[var] for obj in objects],
                                          dim=dimension)

        return new_data

    def data(self):
        for group in self._fields.values():
            for data in group.values():
                yield data

    @classmethod
    def from_xarray(cls, xarray_object):
        """Creates a GeoData object from a xarray.Dataset object.

        Args:
            xarray_object: A xarray.Dataset object.

        Returns:
            GeoData object
        """

        geo_data = cls()
        for var in xarray_object:
            geo_data[var] = xarray_object[var]

        geo_data.attrs.update(**xarray_object.attrs)

        return geo_data

    def get_time_coverage(self, deep=False, to_numpy=False):
        """

        Args:
            deep: Including also subgroups (not only main group).
            to_numpy:

        Returns:

        """
        start = []
        end = []
        for group in self.groups(deep):
            start.append(pd.to_datetime(str(self[group + "time"].min().data)))
            end.append(pd.to_datetime(str(self[group + "time"].max().data)))

        start = min(start)
        end = max(end)

        if to_numpy:
            start = np.datetime64(start)
            end = np.datetime64(end)
        return start, end

    def group(self, name):
        return self._fields[name]

    def groups(self, deep=True):
        """Returns the names of all groups in this GeoData object.

        Args:
            deep: Including also subgroups (not only main group).

        Yields:
            Name of groups.
        """
        if deep:
            for group in self._fields:
                yield group
        else:
            yield "/"

    def items(self, deep=False):
        """

        Args:
            deep: Including also subgroups (not only main group).

        Returns:

        """
        if deep:
            for var in self:
                yield var, self[var]
        else:
            yield from self["/"].items()

    @staticmethod
    def merge(objects):
        """Merge multiple GeoData objects to one.

        Notes:
            Merging of sub groups with the same name does not work properly.

        Args:
            objects: List of GeoData objects.

        Returns:
            A GeoData object.
        """
        merged_data = GeoData()
        for obj in objects:
            # Update the main group
            merged_data.group("/").update(**obj.group("/"))

            sub_groups = obj._fields.copy()
            del sub_groups["/"]

            # Update sub groups
            merged_data._fields.update(**sub_groups)

        return merged_data

    @staticmethod
    def parse(key):
        """Parses *key* into group and field.

        You can access the groups and fields via different keys:

        * "value": Returns ("", "value")
        * "/value": Returns ("", "value")
        * "value1/value2": Returns ("value1", "value2")
        * "value1/": Returns ("value1", "")
        * "/": Returns ("", "")

        Args:
            key:

        Returns:

        """
        if key == "/":
            return "", ""

        if key.startswith("/"):
            return "", key[1:]

        if "/" not in key:
            return "", key

        group, field = key.split("/", 1)
        return group, field

    def plot(self, fields, plot_type="worldmap", fig=None, ax=None, **kwargs):
        """

        Args:
            plot_type:
            fields:
            fig:
            ax:
            **kwargs:

        Returns:

        """

        if plot_type == "worldmap":
            ax, scatter = typhon.plots.worldmap(
                self["lat"],
                self["lon"],
                self[fields[0]],
                fig, ax, **kwargs
            )
        else:
            raise ValueError("Unknown plot type: '{}'".format(plot_type))

        return ax

    def select(self, indices, limit_to=None):
        """Select an TODO.

        Args:
            indices:
            limit_to:

        Returns:

        """
        selected_data = GeoData(self.name+"-selected")

        for field in self:
            group, var = self.parse(field)

            if limit_to is None or group == limit_to:
                selected_data[field] = self[field][indices]

        return selected_data

    def to_xarray(self):
        """Converts this GeoData object to a xarray.Dataset.

        Returns:
            A xarray.Dataset object
        """

        xarray_object = xr.Dataset()
        for var, data in self.items(deep=True):
            xarray_object[var] = data

        xarray_object.attrs.update(**self.attrs)

        start_time, end_time = self.get_time_coverage(deep=True)
        xarray_object.attrs["start_time"] = \
            start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
        xarray_object.attrs["end_time"] = \
            end_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        return xarray_object

    def vars(self, deep=False):
        """Returns the names of all variables in this GeoData object main
        group.

        Args:
            deep: Including also subgroups (not only main group).

        Yields:
            Full name of one variable (including group name).
        """

        for var in self._vars:
            yield var

        if deep:
            for group in self.groups():
                yield self[group].vars(True)
