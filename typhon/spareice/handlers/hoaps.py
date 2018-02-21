import numpy as np

from .common import NetCDF4, expects_file_info

__all__ = [
    'HOAPS',
]


class HOAPS(NetCDF4):
    """File handler that can read data from HOAPS NetCDF4 files.

    This object handles HOAPS V3.3 NetCDF4 files such as they are
    compatible with :mod:`typhon.spareice`, i.e.:

    * Flat all 2-D swath fields into 1-D fields
    * convert the content of the *time* variable to numpydtetime64 objects.

    Examples:



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
    def read(self, file, **kwargs):
        """Reads and parses NetCDF files and load them to an GroupedArrays.

        If you need another return value, change it via the parameter
        *return_type* of the :meth:`__init__` method.

        Args:
            file: Path and name of the file as string or FileInfo object.
            **kwargs: Additional key word arguments that are allowed for the
                :class:`typhon.spareice.handlers.common.NetCDF4` class.

        Returns:
            An GroupedArrays object.
        """

        # import always the standard fields:
        fields = {"time", "lat", "lon"} | set(kwargs.pop("fields", {}))

        # Read the data from the file:
        data = super().read(file, fields=fields, **kwargs)

        for var, var_data in data.items():
            if var == "across_track":
                data[var].attrs["__different_first_dimension__"] = True
            elif len(var_data.shape) == 1:
                data[var] = np.repeat(data[var], 64).flatten()
            elif len(var_data.shape) == 2:
                # Flat all 2-dimensional data
                data[var] = data[var].reshape(
                    -1, data[var].shape[-1]
                )
                data[var] = var_data.flatten()

        return data
