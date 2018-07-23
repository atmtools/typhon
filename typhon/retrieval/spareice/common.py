"""

Examples:

    .. code-block:: python

    from typhon.files import AVHRR_GAC_HDF, CloudSat, FileSet, MHS_HDF
    from typhon.retrieval import SPAREICE

    cloudsat = FileSet(...)
    mhs = FileSet(...)
    avhrr = FileSet(...)

    spareice = SPAREICE(
        file="spareice.json",
    )

    # Either we have already collocated, then we can use the files directly for
    # the training or SPARE-ICE should create the training dataset by itself
    # (should collocate by itself).
    data = spareice.prepare_training_data(
        # Do we have already collocations with all instruments? Put them here:
        collocations=...,
        # OR
        cloudsat=cloudsat, mhs=mhs, avhrr=avhrr,
        # Which time period should be used for training?
        start=..., end=...,
    )

    # To save time and memory space, we can store the current object with
    # the training data to the disk and reuse it later directly. So, we do not
    # have to call spareice.prepare_training_data again:
    data.to_netcdf("spareice_training.nc")

    # Train SPARE-ICE with the data
    spareice.train(data, test_ratio=0.2)

    # After training, we can use the SPARE-ICE retrieval:
    spareice.retrieve(
        # Do we have already collocations with MHS and AVHRR? Put them here:
        collocations=...,
        # Otherwise we can put here each fileset and create collocations
        # on-the-fly
        mhs=mhs, avhrr=avhrr,
        # Which time period should be used for retrieving?
        start=..., end=...,
        output=...,
    )
"""
from os.path import join, dirname

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typhon.collocations import collapse, Collocator
from typhon.utils.timeutils import to_datetime
import xarray as xr

from ..common import RetrievalProduct

__all__ = [
    'SPAREICE',
]

WEIGHTS_DIR = join(dirname(__file__), 'weights')


class SPAREICE:

    def __init__(self, file=None, collocator=None, processes=10, verbose=1):
        if collocator is None:
            self.collocator = Collocator(
                verbose=verbose,
            )
        else:
            self.collocator = collocator

        self.retrieval = RetrievalProduct(
            parameters_file=file, n_jobs=processes,
            scaler="robust", verbose=verbose
        )

        self.verbose = verbose
        self.name = "SPARE-ICE"

    def _debug(self, msg):
        if self.verbose > 1:
            print(f"[{self.name}] {msg}")

    def info(self, msg):
        if self.verbose > 0:
            print(f"[{self.name}] {msg}")

    def load_training(self, filename):
        self.retrieval.load_parameters(filename)

    def save_training(self, filename):
        self.retrieval.save_parameters(filename)

    @staticmethod
    def get_training_data(
            collocations=None, cloudsat=None, mhs=None,
            avhrr=None, start=None, end=None
    ):

        # We need all original data (cloudsat, mhs, avhrr) if we do not have
        # collocations:
        if (cloudsat is None or mhs is None or avhrr is None) \
                and collocations is None:
            raise ValueError(
                "If no collocations are given, you need to pass the original "
                "data of MHS and AVHRR!"
            )

        #
        if cloudsat is None:
            # The user has given some collocations to us, use them!
            data = xr.concat(collocations[start:end], dim="collocation")
        else:
            print("No collocations given or no collocations found in the given"
                  "time period. Start collocation toolkit to find them.")
            collocations.search(
                [mhs, cloudsat], start=start, end=end, processes=1,
                max_interval="5 min", max_distance="7.5 km", verbose=2,
            )
            ...

        return data

    @staticmethod
    def get_inputs(data):
        """Get the input fields for SPARE-ICE training / retrieval"""

        fields = {
            "mhs_channel3": data["Data_btemps"].isel(
                **{"channel": 2}
            ),
            "mhs_channel4": data["Data_btemps"].isel(
                **{"channel": 3}
            ),
            "mhs_channel5": data["Data_btemps"].isel(
                **{"channel": 4}
            ),
            "lat": data["lat"],
            "mhs_scnpos": data["scnpos"],
            "satellite_azimuth_angle":
                data["Geolocation_Satellite_azimuth_angle"],
            "satellite_zenith_angle":
                data["Geolocation_Satellite_zenith_angle"],
            "solar_azimuth_angle":
                data["Geolocation_Solar_azimuth_angle"],
            "solar_zenith_angle":
                data["Geolocation_Solar_zenith_angle"],
            "relative_azimuth_angle":
                data["AVHRR/Geolocation_Relative_azimuth_angle_mean"],
            "avhrr_channel1": data["AVHRR/Data_btemps_mean"].isel(
                **{"channel": 0}
            ),
            "avhrr_channel2": data["AVHRR/Data_btemps_mean"].isel(
                **{"channel": 1}
            ),
            "avhrr_channel4": data["AVHRR/Data_btemps_mean"].isel(
                **{"channel": 3}
            ),
            "avhrr_channel5": data["AVHRR/Data_btemps_mean"].isel(
                **{"channel": 4}
            ),
            "avhrr_scnpos": data["AVHRR/scnpos_mean"],
        }

        return pd.DataFrame(fields)

    @staticmethod
    def get_targets(data):
        """Get the target fields for SPARE-ICE training"""
        targets = pd.DataFrame({
            "iwp_log10": np.log10(data["2C-ICE_ice_water_path_mean"])
        })
        return targets

    @staticmethod
    def split_data(data, ratio=None, shuffle=True):
        indices = np.arange(data.lat.size)
        if shuffle:
            np.random.shuffle(indices)

        dim = data.lat.dims[0]
        boundary = int(data.lat.size * ratio)

        return (
            data.isel(**{dim: indices[:boundary]}),
            data.isel(**{dim: indices[boundary:]}),
        )

    def train(self, data, test_ratio=None):

        inputs = self.get_inputs(data)
        targets = self.get_targets(data)

        # Make the training and testing data:
        inputs_train, inputs_test, targets_train, targets_test \
            = train_test_split(inputs, targets, test_size=test_ratio)

        self.info("Train SPARE-ICE")
        train_score = self.retrieval.train(
            inputs_train,
            targets_train,
        )
        self.info(f"Training score: {train_score:.2f}")

        test_score = self.retrieval.score(inputs_test, targets_test)
        self.info(f"Testing score: {test_score:.2f}")

    def _get_retrieval_data(self, collocations, mhs, avhrr, start, end,
                            processes):
        # We need all original data (mhs, avhrr) if we do not have
        # collocations:
        if (mhs is None or avhrr is None) and collocations is None:
            raise ValueError(
                "If no collocations are given, you need to pass the original "
                "data of MHS and AVHRR!"
            )

        if mhs is None:
            yield from collocations.icollect(start=start, end=end)
        else:
            data_iterator = self.collocator.collocate_filesets(
                [avhrr, mhs], start=start, end=end, processes=processes,
                max_interval="30s", max_distance="7.5 km",
            )
            for data, attributes in data_iterator:
                yield collapse(data, reference="MHS"), attributes

                # Shall we save the collocations to disk?
                if collocations is None:
                    continue

                filename = collocations.get_filename(
                    [to_datetime(data.attrs["start_time"]),
                     to_datetime(data.attrs["end_time"])],
                    fill=attributes
                )

                # Write the data to the file.
                self._debug(f"Store collocations to \n{filename}")
                collocations.write(data, filename)

    def retrieve(self, data, from_collocations=False, as_log10=False):
        """Retrieve SPARE-ICE for the input variables

        Args:
            data: A pandas.DataFrame object with required input fields (see
                above) or a xarray.Dataset if `from_collocations` is True.
            from_collocations: Set this to true if `data` comes from
                collocations and the fields will be correctly transformed.
            as_log10: If true, the retrieved IWP will be returned as logarithm
                of base 10.

        Returns:
            A pandas DataFrame object with the retrieved IWP.
        """
        # use the standard weights of SPARE-ICE:
        if not self.retrieval.is_trained():
            try:
                self.retrieval.load_parameters(
                    join(WEIGHTS_DIR, "standard.json")
                )
            except Exception as e:
                print("Could not load the standard weights of SPARE-ICE!")

        # We have to rename the variables when they come from collocations:
        if from_collocations and isinstance(data, xr.Dataset):
            inputs = self.get_inputs(data)
        elif isinstance(data, pd.DataFrame):
            inputs = data
        else:
            if from_collocations:
                raise ValueError("data must be a xarray.Dataset!")
            else:
                raise ValueError(
                    "data must be a pandas.DataFrame! You can also set "
                    "from_collocations to True and use a xarray.Dataset.")

        retrieved = self.retrieval.retrieve(inputs)
        if not as_log10:
            retrieved.rename(columns={"iwp_log10": "iwp"}, inplace=True)
            retrieved["iwp"] = 10**retrieved["iwp"]

        return retrieved

    def retrieve_from_filesets(
            self, collocations=None, mhs=None, avhrr=None,
            output=None, start=None, end=None, processes=None,
    ):
        """Retrieve SPARE-ICE from all files in a fileset

        You can use this either with already collocated MHS and AVHRR data
        (then use the parameter `collocations`) or can let MHS and AVHRR
        collocate on-the-fly by passing the filesets with the raw data (use
        `mhs` and `avhrr` then).

        Args:
            collocations:
            mhs:
            avhrr:
            output:
            start:
            end:
            processes:

        Returns:
            None
        """
        if processes is None:
            processes = 1

        data_iterator = self._get_retrieval_data(
            collocations, mhs, avhrr, start, end, processes
        )

        for data, attributes in data_iterator:
            retrieved = self.retrieve(data, from_collocations=True)
            retrieved = retrieved.to_xarray()
            retrieved.rename({"index": "collocation"}, inplace=True)

            # Add more information:
            retrieved["iwp"].attrs = {
                "units": "g/m^2",
                "name": "Ice Water Path",
                "description": "Ice Water Path retrieved from SPARE-ICE"
            }
            retrieved["lat"] = data["lat"]
            retrieved["lon"] = data["lon"]
            retrieved["time"] = data["time"]

            filename = output.get_filename(
                [to_datetime(data.attrs["start_time"]),
                 to_datetime(data.attrs["end_time"])], fill=attributes
            )

            # Write the data to the file.
            self._info(f"Store SPARE-ICE to \n{filename}")
            output.write(retrieved, filename)
