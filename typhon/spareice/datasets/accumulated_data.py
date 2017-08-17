__all__ = [
    'AccumulatedData',
]

import matplotlib.pyplot as plt
import typhon.plots
import xarray as xr


class AccumulatedData(xr.Dataset):
    markers = ["o", "+", "*", "x", ".", ">"]
    colors = ["r", "b", "y", "g", "k"]

    def __init__(self, *args, **kwargs):
        """

        Args:
            *args:
            **kwargs:
        """
        super(AccumulatedData, self).__init__(*args, **kwargs)

    @classmethod
    def from_xarray(cls, xarray_object):
        """

        Args:
            xarray_object:

        Returns:

        """
        accumulated_data = cls()

        # Is this good style?
        accumulated_data.__dict__ = xarray_object.__dict__
        return accumulated_data

    @staticmethod
    def merge(objects):
        """

        Args:
            objects:

        Returns:

        """
        data = AccumulatedData()
        names = []
        for i, obj in enumerate(objects):
            name = obj.attrs["name"]
            names.append(name)
            renames = {old_name : name+"."+old_name for old_name in obj}
            data = xr.merge([data, obj.rename(renames)])

        data.attrs["datasets"] = names

        return AccumulatedData.from_xarray(data)

    def has_fields(self, *fields):
        """

        Args:
            *fields:

        Returns:

        """
        for field in fields:
            if not field in self:
                return False
        return True

    def plot(self, plot_type, fields=(), fig=None, ax=None, **kwargs):
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
            if self.has_fields(*fields):
                ax = typhon.plots.worldmap(self[fields[0]], self[fields[1]], self[fields[2]], fig, ax, **kwargs)
                #plt.legend([fields[2]])
            elif "datasets" in self.attrs:
                # This AccumulatedData object contains data from multiple datasets.
                legend = []
                for i, dataset in enumerate(self.attrs["datasets"]):
                    kwargs["marker"] = AccumulatedData.markers[i]

                    variable = self[dataset+"."+fields[2]] if len(fields) == 3 else AccumulatedData.colors[i]

                    ax, scatter = typhon.plots.worldmap(
                        self[dataset+"."+fields[0]],
                        self[dataset+"."+fields[1]],
                        variable,
                        fig, ax, **kwargs)

                    legend.append(dataset)

                plt.legend(legend)
            else:
                raise ValueError("This AccumulatedData object needs fields such as lat, lon and time!")

            return ax
        elif plot_type == "collocations":
            if fields and len(self.attrs["datasets"]) == 2:
                plt.grid()
                dataset1, dataset2 = self.attrs["datasets"]
                scatter_plot = plt.scatter(
                    self[dataset1 + "." + fields[0]][self[dataset1 + ".collocation_indices"]],
                    self[dataset2 + "." + fields[0]][self[dataset2 + ".collocation_indices"]],
                    **kwargs)
                plt.xlabel(dataset1 + "." + fields[0])
                plt.ylabel(dataset2 + "." + fields[0])
            else:
                legend = []
                for i, dataset in enumerate(self.attrs["datasets"]):
                    kwargs["marker"] = AccumulatedData.markers[i]

                    ax, scatter = typhon.plots.worldmap(
                        self[dataset + ".lat"][self[dataset + ".collocation_indices"]],
                        self[dataset + ".lon"][self[dataset + ".collocation_indices"]],
                        range(len(self[dataset + ".collocation_indices"])),
                        fig, ax, **kwargs)

                    legend.append(dataset)

                plt.legend(legend)

    def to_xarray(self):
        """

        Returns:

        """
        obj = xr.Dataset()
        obj.__dict__ = self.__dict__
        return obj