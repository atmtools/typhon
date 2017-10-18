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
    def __init__(self, name=None):
        """A specialised container for geographical indexed data (with
        longitude, latitude and time field).

        Since the xarray.Dataset class cannot handle groups and is difficult to
        extend (see http://xarray.pydata.org/en/stable/internals.html#extending-xarray
        for more details).


        Args:
            data: (optional) A xr.Dataset object, must contain a *time*,
                *lat* and *lon* field.
            prefix: (optional) The fields might have a prefix before their
                name (e.g. *DatasetA.time*).
        """

        self._groups = defaultdict(GeoData)
        # Add the main group:
        self._groups["/"] = {}

        if name is None:
            self.name = "<GeoData="+str(id(self))+">"
        else:
            self.name = name

    def __contains__(self, item):
        group, field = self.parse(item)
        if group not in self._groups:
            return False

        return field in self._groups[group]

    def __iter__(self):
        self._vars = self.vars(recursive=True)
        return self

    def __next__(self):
        return next(self._vars)

    def __len__(self):
        if "time" not in self:
            return 0

        return len(self["time"])

    def __str__(self):
        info = self.name + "\n"
        for group in self.groups():
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
        if not field:
            del self._groups[group]
        else:
            del self._groups[group][field]

    def __getitem__(self, item):
        group, field = self.parse(item)
        if not field:
            return self._groups[group]
        return self._groups[group][field]

    def __setitem__(self, key, value):
        group, field = self.parse(key)
        if not field:
            if not isinstance(value, GeoData):
                raise ValueError("Group value must be a GeoData object!")
            elif group == "/" and not isinstance(value, dict):
                raise ValueError("Main group value must be a dictionary "
                                 "or xarray.DataArray!")
            self._groups[group] = value
        else:
            self._groups[group][field] = value

    def bin(self, bins, fields=None):
        new_data = GeoData()
        for var, data in self.group("/").items():
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

    def collapse(self, bins, collapser=np.nanmean,
                 variation_filter=None):
        # Exclude all bins where the inhomogeneity (variation) is too high
        passed = np.ones_like(bins).astype("bool")
        if isinstance(variation_filter, tuple):
            if len(variation_filter) >= 2:
                if len(self[variation_filter[0]].dims) > 1:
                    raise ValueError("The variation filter can only be used "
                                     "for 1-dimensional data! I.e. the field "
                                     "'{}' must be 1-dimensional!".format(
                                        variation_filter[0]))
                binned = self.bin(bins, fields=(variation_filter[0]))
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
        collapsed_data = self.bin(bins[passed])
        for var, data in collapsed_data.items():
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
        for group in self._groups.values():
            for data in group.values():
                yield data

    @classmethod
    def from_xarray(cls, xarray_object, mapping=None, prefix=None):
        """Creates a GeoData object from a xarray.Dataset object.

        Args:
            xarray_object:
            mapping:
            prefix:

        Returns:
            GeoData object
        """

        if prefix is not None:
            # Retrieve only those field which start with the searched prefix:
            fields_to_drop = [
                field for field in xarray_object
                if not field.startswith(prefix)]
            xarray_object = xarray_object.drop(fields_to_drop)

        return cls(xarray_object, prefix)

    def get_time_coverage(self, to_numpy=False):
        start = pd.to_datetime(str(self["time"].min().data))
        end = pd.to_datetime(str(self["time"].max().data))

        if to_numpy:
            start = np.datetime64(start)
            end = np.datetime64(end)
        return start, end

    def group(self, name):
        return self._groups[name]

    def groups(self):
        """Returns the names of all groups in this GeoData object.

        Yields:
            Name of groups.
        """
        for group in self._groups:
            yield group

    def items(self, recursive=False):
        if recursive:
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

            sub_groups = obj._groups.copy()
            del sub_groups["/"]

            # Update sub groups
            merged_data._groups.update(**sub_groups)

        return merged_data

    @staticmethod
    def parse(field):
        if field == "/":
            return "/", ""

        if "/" not in field:
            return "/", field

        group, key = field.split("/", 1)
        return group, key

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
            dimension:
            limit_to:

        Returns:

        """
        selected_data = GeoData(self.name+"-selected")

        for field in self:
            group, var = self.parse(field)

            print("Group: ", group, ", Var:", var)

            if limit_to is None or group == limit_to:
                print(self[field])
                print(indices)
                selected_data[field] = self[field][indices]

        return selected_data

    def to_xarray(self):
        """

        Returns:
            xarray.Dataset()
        """
        xarray_object = self.data

        start_time, end_time = self.get_time_coverage()
        xarray_object.attrs["start_time"] = \
            start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
        xarray_object.attrs["end_time"] = \
            end_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        return xarray_object

    def vars(self, recursive=False):
        """Returns the names of all variables in this GeoData object main
        group.

        Args:
            recursive: Include all variables of its subgroups.

        Yields:
            Full name of one variable (including group name).
        """

        for group in self.groups():
            if group == "/":
                yield from self.group(group)
            else:
                for var in self.group(group).vars():
                    if group == "/":
                        yield var
                    else:
                        yield group+"/"+var
