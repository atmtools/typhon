import xarray as xr

from .common import NetCDF4, expects_file_info

__all__ = [
    'HOAPS',
]


class HOAPS(NetCDF4):
    """File handler that can read data from HOAPS NetCDF4 files.

    This object handles HOAPS V3.3 NetCDF4 files such as they are
    compatible with :mod:`typhon.collocations`, i.e.:

    * convert the content of the *time* variable to numpydtetime64 objects.

    Examples:

        Draw a world map with all measurements of OceanRAIN:

        .. :code-block:: python3

            from typhon.files import FileSet, HOAPS

            hoaps = FileSet(
                name="HOAPS",
                path=".../HOAPS_v3.3/{year}-{month}/hoaps-s.f16.{year}-{month}-{day}.nc",
                handler=HOAPS(),
                # This extracts the field *asst*:
                read_args={
                    "fields": ["asst"],
                },
                # The path of HOAPS files does not provide the end of the
                # file's time coverage. Hence, we set it here explicitly:
                time_coverage="1 day",
            )

            print(hoaps["2013"])

    """

    def __init__(self, **kwargs):
        """Initializes a HOAPS file handler class

        Args:
            **kwargs: Additional key word arguments that are allowed for the
                :class:`typhon.files.handlers.common.NetCDF4` class.

        """
        # Call the base class initializer
        super().__init__(**kwargs)

    @expects_file_info()
    def get_info(self, file, **kwargs):
        """Get information about an HOAPS dataset file

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
            fields = {"time", "lat", "lon"} | set(fields)

        # xarray has problems with decoding the time variable correctly. Hence,
        # we disable it here:
        decode_cf = kwargs.pop("decode_cf", True)

        data = super().read(filename, fields=fields, decode_cf=False, **kwargs)

        # Then we fix the problem (we need integer64 instead of integer 32):
        attrs = data["time"].attrs.copy()
        data["time"] = data["time"].astype(int)
        data["time"].attrs = attrs

        # Do decoding now (just if the user wanted it!)
        if decode_cf:
            return xr.decode_cf(data)

        return data
