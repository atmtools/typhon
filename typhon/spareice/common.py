"""

Examples:

    .. code-block:: python

    from typhon.files import AVHRR_GAC_HDF, CloudSat, FileSet, MHS_HDF
    from typhon.spareice import SPAREICE

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
        # and set collocate to False
        collocate=False,
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
        collocate=False,
        # Otherwise we can put here each fileset and create collocations
        # on-the-fly
        mhs=mhs, avhrr=avhrr,
        # Which time period should be used for retrieving?
        start=..., end=...,

    )
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typhon.collocations import Collocator
from typhon.retrieval import RetrievalProduct
import xarray as xr


class SPAREICE:

    def __init__(self, file=None, verbose=1):
        self.retrieval = RetrievalProduct(
            parameters_file=file,
            #verbose=verbose
        )
        self.verbose = verbose

    def debug(self, msg):
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
    def get_training_collocations(
        collocations, cloudsat, mhs, avhrr, start, end
    ):
        if collocations is None:
            # The user does not want to save the collocations to disk
            ...
        else:
            # The user wants to save the collocations to disk
            # Search for collocations and store them
            collocations.search(
                [mhs, cloudsat], start=start, end=end, processes=1,
                max_interval="5 min", max_distance="7.5 km", verbose=2,
            )

        return data

    @staticmethod
    def prepare_training_data(
        collocations=None, cloudsat=None, mhs=None, avhrr=None,
        start=None, end=None
    ):

        # We should examine this more deeply. What if `collocations`is given
        # but empty?
        collocations_found = collocations is not None

        if collocations_found:
            print("Found collocations for training data")
            data = xr.concat(collocations[start:end], dim="collocation")
        else:
            print("No collocations given or no collocations found in the given"
                  "time period. Start collocation toolkit to find them.")
            data = SPAREICE.get_training_collocations(
                collocations, cloudsat, mhs, avhrr, start, end
            )

        # Give more meaningful names to the variables
        mapping = {
            "Data_btemps": "MHS_BT",
            "scnpos": "MHS_scnpos",
            "MHS_2C-ICE/channel": "MHS_channel",
            "Geolocation_Satellite_azimuth_angle": "satellite_azimuth_angle",
            "Geolocation_Satellite_zenith_angle": "satellite_zenith_angle",
            "Geolocation_Solar_azimuth_angle": "solar_azimuth_angle",
            "Geolocation_Solar_zenith_angle": "solar_zenith_angle",
            "AVHRR/Geolocation_Relative_azimuth_angle_mean":
                "relative_azimuth_angle",
            "2C-ICE_ice_water_path_mean": "IWP",
            "2C-ICE_ice_water_path_std": "IWP_std",
            "2C-ICE_ice_water_path_number": "IWP_number",
            "AVHRR/Data_btemps_mean": "AVHRR_BT",
            "AVHRR/Data_btemps_std": "AVHRR_std",
            "AVHRR/Data_btemps_number": "AVHRR_number",
            "AVHRR/scnpos_mean": "AVHRR_scnpos",
            "AVHRR/channel": "AVHRR_channel",
        }
        data.rename(mapping, inplace=True)

        return data

    def train(self, data, test_ratio=None):

        # Make the training and testing data:
        inputs = pd.DataFrame({
            "mhs_channel3": data["MHS/Data_btemps"].isel(channel=2),
            "mhs_channel4": data["MHS/Data_btemps"].isel(channel=3),
            "mhs_channel5": data["MHS/Data_btemps"].isel(channel=4),
            "lat": data["lat"],
            "mhs_scnpos": data["MHS/scnpos"],
        })
        targets = pd.DataFrame({
            "IWP_log10": np.log10(data["IWP"])
        })
        inputs_train, inputs_test, targets_train, targets_test \
            = train_test_split(inputs, targets)

        print("Train SPARE-ICE")
        # If we have not trained it already, let's train it here:
        train_score = retriever.train(
            # The input fields for the training. The keys are the names of the input
            # fields, the values define where the values for this input come from.
            inputs_train,
            # The target fields for the training. The keys are the names of the
            # target/output fields, the values define where the values for this target
            # come from.
            targets_train,
            verbose=2,
        )
            print(f"Training score: {train_score:.2f}")
        test_score = retriever.score(inputs_test, targets_test)
        print(f"Testing score: {test_score:.2f}")

    def _get_retrieve_data(self, collocations, mhs, avhrr, output, start, end):
        yield None

    def _prepare_retrieve_inputs(self, data):
        inputs = pd.DataFrame({
            "mhs_channel3": data["MHS/Data_btemps"].isel(channel=2),
            "mhs_channel4": data["MHS/Data_btemps"].isel(channel=3),
            "mhs_channel5": data["MHS/Data_btemps"].isel(channel=4),
            "lat": data["lat"],
            "mhs_scnpos": data["MHS/scnpos"],
        })

        return inputs

    def retrieve(
            self, collocations=None, mhs=None, avhrr=None, output=None,
            start=None, end=None
    ):

        data_iterator = self._get_retrieve_data(
            collocations, mhs, avhrr, output, start, end
        )

        for data in data_iterator:
            inputs = self._prepare_retrieve_inputs(data)

            retrieved = self.retrieval.run(inputs)

            # Add geolocation and time information:
            retrieved["lat"] = data["lat"]
            retrieved["lon"] = data["lon"]
            retrieved["time"] = data["time"]

            # Store the output:
            output[date1:date2] = retrieved

