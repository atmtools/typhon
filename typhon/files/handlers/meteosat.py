from datetime import datetime, timedelta
from os.path import dirname, join
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import xarray as xr

from .common import expects_file_info, HDF5


__all__ = [
    'SEVIRI',
]


class SEVIRI(HDF5):
    """File handler for SEVIRI level 1.5 HDF files
    """

    # Cache the grid of SEVIRI since it is expensive to calculate
    _grid = None

    channel_names = {
        'channel_1': 'VIS006',
        'channel_2': 'VIS008',
        'channel_3': 'IR_016',
        'channel_4': 'IR_039',
        'channel_5': 'WV_062',
        'channel_6': 'WV_073',
        'channel_7': 'IR_087',
        'channel_8': 'IR_097',
        'channel_9': 'IR_108',
        'channel_10': 'IR_120',
        'channel_11': 'IR_134',
        'channel_12': 'HRV'
    }

    def __init__(self, **kwargs):
        """

        Args:
            **kwargs: Additional key word arguments for base class.
        """

        # Call the base class initializer
        super().__init__(**kwargs)

        # We are going to import those fields per default (not channel 12
        # because it has a different resolution):
        self.standard_fields = {
            f"U-MARF/MSG/Level1.5/DATA/Channel {ch:02d}/IMAGE_DATA"
            for ch in range(1, 12)
        }

        # Standard fields can be overwritten by the user. However, we always
        # need the field with the scale and offset parameters for the channels:
        self.mandatory_fields = {
            "U-MARF/MSG/Level1.5/METADATA/HEADER/RadiometricProcessing/Level15ImageCalibration_ARRAY"  # noqa
        }

        # Map the standard fields to standard names (make also the names of all
        # dimensions more meaningful):
        self.mapping = {
            f"U-MARF/MSG/Level1.5/DATA/Channel {ch:02d}/IMAGE_DATA":
                f"channel_{ch}"
            for ch in range(1, 12)
        }
        self.reverse_mapping = {
            self.channel_names[value]: key
            for key, value in self.mapping.items()
        }
        self.mapping["U-MARF/MSG/Level1.5/METADATA/HEADER/RadiometricProcessing/Level15ImageCalibration_ARRAY"] = "counts_to_rad"  # noqa

    @staticmethod
    def grid():
        """Return the geo-grid of the SEVIRI images (lat and lon)

        Notes:
            When calling this method for the first time, the latitudes and
            longitudes are going to be calculated. This may take a while. The
            results is cached, so multiple calls are faster. The caching may
            need a lot of space (the grid is HUGE)!

        Returns:
            A xarray.Dataset with lat and lon fields
        """
        if SEVIRI._grid is not None:
            return SEVIRI._grid

        # The SEVIRI image has a size of 3712x3712 pixels
        height_width = np.arange(3712)

        # Get the intermediate coordinates
        OFF = 1856
        FAC = -781648343
        intermediate_coords = 2 ** 16 * (height_width + 1 - OFF) / FAC

        # And all their combinations
        x, y = np.meshgrid(intermediate_coords, intermediate_coords)

        # Calculate the geo locations (lat, lon) from them
        cos_x = np.cos(x)
        cos_y = np.cos(y)
        sin_y = np.sin(y)

        sa = (42164 * cos_x * cos_y) ** 2 \
            - (cos_y ** 2 + 1.006803 * sin_y ** 2) * 1737121856

        sd = np.sqrt(sa)
        sn = (42164 * cos_x * cos_y - sd) / (
                cos_y ** 2 + 1.006803 * sin_y ** 2)
        s1 = 42164 - sn * cos_x * cos_y
        s2 = sn * np.sin(x) * cos_y
        s3 = -1 * sn * sin_y
        sxy = np.sqrt(s1 ** 2 + s2 ** 2)

        # conversion to degree
        lon = np.arctan(s2 / s1) * 180 / np.pi
        lat = np.arctan(1.006803 * (s3 / sxy)) * 180 / np.pi

        # We cache the grid
        ds = xr.Dataset({
            "lat": (("line", "column"), -lat),
            "lon": (("line", "column"), -lon),
        })

        ds["lat"].attrs = {
            "long_name": "latitude",
            "units": "degrees [-90, 180]",
        }
        ds["lon"].attrs = {
            "long_name": "longitude",
            "units": "degrees [-180, 180]",
        }

        SEVIRI._grid = ds

        return SEVIRI._grid

    @expects_file_info()
    def read(self, file_info, fields=None, calibration=True, **kwargs):
        """Read SEVIRI HDF5 files and load them to a xarray.Dataset

        Args:
            file_info: Path and name of the file as string or FileInfo object.
                This can also be a tuple/list of file names or a path with
                asterisk.
            fields: Field names that you want to extract from this file as a
                list.
            **kwargs: Additional keyword arguments that are valid for
                :class:`typhon.files.handlers.common.NetCDF4`.

        Returns:
            A xrarray.Dataset object.
        """

        # Here, the user fields overwrite the standard fields:
        if fields is None:
            fields = self.standard_fields | self.mandatory_fields
        else:
            fields = self.mandatory_fields | {
                self.reverse_mapping[field]
                for field in fields
                if field not in ["time", "lat", "lon"]
            }

        # We catch the user mapping here, since we do not want to deal with
        # user-defined names in the further processing. Instead, we use our own
        # mapping at the moment, and apply the user mapping later.
        user_mapping = kwargs.pop("mapping", None)

        dataset = super(SEVIRI, self).read(
            file_info, fields=fields, mapping=self.mapping
        )

        # We set the names of each channel variable explicitly:
        for name, var in dataset.variables.items():
            if name not in ["time", "lat", "lon", "counts_to_rad"]:
                dataset[name] = ["line", "column"], var.values

        # Convert the counts to brightness temperatures:
        if calibration:
            dataset = self.counts_to_bt(dataset)

        # Add the latitudes and longitudes of the grid points:
        dataset = xr.merge([dataset, self.grid()])

        # Add the time variable (is derived from the filename normally):
        if file_info.times[0] is None:
            warnings.warn(
                "SEVIRI: The time field was not specified by the filename! Set"
                "it to 1970-01-01!"
            )
            dataset["time"] = ("time", np.array(["1970-01-01"], dtype="M8[s]"))
        else:
            dataset["time"] = ("time", [file_info.times[0]])

        # For collocating and other things, we always need the three dimensions
        # time, lat and lon to be "connected". Hence, add the time dimension to
        # each data variable as extra dimension.
        dataset = xr.concat([dataset], dim="time")

        # We want the final field names to be meaningful
        mapping = {
            old: new
            for old, new in self.channel_names.items()
            if old in dataset.variables
        }
        dataset = dataset.rename(mapping)
        return dataset

    @staticmethod
    def counts_to_bt(dataset):
        """Convert the counts to brightness temperatures

        Cold channels (4, 5, 6, 7, 8, 9, 10, 11) are converted to brightness
        temperatures while warm channels are converted to radiances.

        Args:
            dataset: A xarray.Dataset with counts

        Returns:
            A xarray.Dataset with brightness temperatures.
        """
        # We have to convert in two steps:
        # 1) Convert the counts into radiances with the given offset and scale
        # parameters (these are stored in the SEVIRI HDF file).
        # 2) Convert the radiances into brightness temperatures (only
        # applicable to the warm channels).
        conversion_table = pd.read_csv(
            join(dirname(__file__), "seviri_radiances_to_bt.csv")
        )
        for var in dataset.data_vars:
            if not var.startswith("channel_"):
                continue

            channel = int(var.split("_")[1])
            coeffs = dataset["counts_to_rad"][channel-1].item(0)
            dataset[f"channel_{channel}"] = \
                coeffs[0]*dataset[f"channel_{channel}"] + coeffs[1]

            if channel < 4:
                continue

            # Build the converter function
            converter = interp1d(
                conversion_table[f"ch{channel}"], conversion_table["BT"],
                fill_value=np.nan, bounds_error=False
            )
            dataset[f"channel_{channel}"] = xr.DataArray(
                converter(dataset[f"channel_{channel}"]),
                dims=dataset[f"channel_{channel}"].dims
            )

        # Drop the conversion variable:
        return dataset.drop("counts_to_rad")
