from .common import NetCDF4, expects_file_info

__all__ = [
    'OceanRAIN',
]


class OceanRAIN(NetCDF4):
    """File handler that can read data from OceanRAIN NetCDF4 files.

    This object handles OceanRAIN V1.0 NetCDF4 files such as they are
    compatible with :mod:`typhon.collocations`, i.e.:

    * rename *latitude* and *longitude* field to *lat* and *lon*.
    * convert the content of the *time* variable to numpydtetime64 objects.

    Examples:

        Draw a world map with all measurements of OceanRAIN:

        .. :code-block:: python3

            from typhon.files import FileSet, OceanRAIN
            from typhon.plots import worldmap

            # Create a Dataset object that points to the files:
            ocean_rain = FileSet(
                "OceanRAIN_1.0/OceanRAIN__W__{ship}_{ship_id}__UHAM-ICDC__v1_0.nc",
                handler=OceanRAIN(),
                info_via="both",
            )

            # Create a figure:
            fig = plt.figure(figsize=(12, 12))
            ax = None

            # This iterates through all files from OceanRAIN and plots them:
            for file, data in ocean_rain.icollect(return_info=True):
                label = f"{file.attr['ship']} ({data['lat'].size} mins)"

                # We only plot every 1000th point to draw the worldmap faster:
                ax, plot = worldmap(
                    data["lat"][::1000], data["lon"][::1000], ax=ax,
                    s=6, label=label, background=True
                )

            # Add a nice legend
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles, labels, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=2, borderaxespad=0., #mode="expand",
            )
            fig.show()

    """

    def __init__(self, **kwargs):
        """Initializes a OceanRAIN file handler class.

        Args:
            **kwargs: Additional key word arguments that are allowed for the
                :class:`typhon.spareice.handlers.common.NetCDF4` class.

        """
        # Call the base class initializer
        super().__init__(**kwargs)

    @expects_file_info()
    def get_info(self, file, **kwargs):
        """Get information about an OceanRAIN dataset file

        Args:
            file: A string containing path and name or a :class:`FileInfo`
                object of the file of which to get the information about.
            **kwargs: Additional keyword arguments.

        Returns:
            A :class:`FileInfo` object.
        """
        data = super().read(file, fields=("time",))

        file.times = [data["time"].min().item(0), data["time"].max().item(0)]

        return file

    @expects_file_info()
    def read(self, filename, **kwargs):
        """Read and parse a NetCDF file and load it to a xarray.Dataset

        Args:
            filename: Path and name of the file as string or FileInfo object.
            **kwargs: Additional key word arguments that are allowed for the
                :class:`~typhon.files.handlers.common.NetCDF4` class.

        Returns:
            A xarray.Dataset object.
        """

        # Make sure that the standard fields are always gonna be imported:
        fields = kwargs.pop("fields", None)
        if fields is not None:
            fields = {"time", "latitude", "longitude"} | set(fields)

        # This renaming makes the data compatible for collocate routines:
        mapping = {
            "latitude": "lat",
            "longitude": "lon",
            **kwargs.get("mapping", {})
        }

        return super().read(filename, fields=fields, mapping=mapping, **kwargs)
