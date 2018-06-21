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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typhon.collocations import collapse, Collocator
from typhon.retrieval import RetrievalProduct
from typhon.utils.timeutils import to_datetime
import xarray as xr


__all__ = [
    'SPAREICE',
]


class SPAREICE:

    def __init__(self, file=None, collocator=None, verbose=1):
        if collocator is None:
            self.collocator = Collocator(
                verbose=verbose,
            )
        else:
            self.collocator = collocator

        self.retrieval = RetrievalProduct(
            parameters_file=file,
            verbose=verbose
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
    def _get_inputs(data):
        """Get the input fields for SPARE-ICE training / retrieval"""
        print(data)

        inputs = pd.DataFrame({
            "mhs_channel3": data["Data_btemps"].isel(
                **{"MHS/channel": 2}
            ),
            "mhs_channel4": data["Data_btemps"].isel(
                **{"MHS/channel": 3}
            ),
            "mhs_channel5": data["Data_btemps"].isel(
                **{"MHS/channel": 4}
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
                **{"AVHRR/channel": 1}
            ),
            "avhrr_channel2": data["AVHRR/Data_btemps_mean"].isel(
                **{"AVHRR/channel": 2}
            ),
            "avhrr_channel3": data["AVHRR/Data_btemps_mean"].isel(
                **{"AVHRR/channel": 3}
            ),
            "avhrr_channel4": data["AVHRR/Data_btemps_mean"].isel(
                **{"AVHRR/channel": 4}
            ),
            "avhrr_channel5": data["AVHRR/Data_btemps_mean"].isel(
                **{"AVHRR/channel": 5}
            ),
            "avhrr_scnpos": data["AVHRR/scnpos_mean"],
        })

        return inputs

    @staticmethod
    def _get_targets(data):
        """Get the target fields for SPARE-ICE training"""
        targets = pd.DataFrame({
            "IWP_log10": np.log10(data["2C-ICE_ice_water_path_mean"])
        })
        return targets

    def train(self, data, test_ratio=None):

        inputs = self._get_inputs(data)
        targets = self._get_targets(data)

        # Make the training and testing data:
        inputs_train, inputs_test, targets_train, targets_test \
            = train_test_split(inputs, targets, test_size=test_ratio)

        self.info("Train SPARE-ICE")
        # If we have not trained it already, let's train it here:
        train_score = self.retrieval.train(
            inputs_train,
            targets_train,
        )
        self.info(f"Training score: {train_score:.2f}")

        test_score = self.retrieval.score(inputs_test, targets_test)
        self.info(f"Testing score: {test_score:.2f}")

    def _get_retrieval_data(self, collocations, mhs, avhrr, start, end):
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
                [avhrr, mhs], start=start, end=end, processes=10,
                max_interval="5 min", max_distance="7.5 km",
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
                self.debug(f"Store collocations to \n{filename}")
                collocations.write(data, filename)

    def retrieve(
            self, collocations=None, mhs=None, avhrr=None,
            output=None, start=None, end=None,
    ):

        data_iterator = self._get_retrieval_data(
            collocations, mhs, avhrr, start, end
        )

        for data, attributes in data_iterator:
            inputs = self._get_inputs(data)

            retrieved = self.retrieval.retrieve(inputs)

            # Add geolocation and time information:
            retrieved["lat"] = data["lat"]
            retrieved["lon"] = data["lon"]
            retrieved["time"] = data["time"]

            filename = output.get_filename(
                [to_datetime(data.attrs["start_time"]),
                 to_datetime(data.attrs["end_time"])], fill=attributes
            )

            # Write the data to the file.
            self.debug(f"Store SPARE-ICE to \n{filename}")
            output.write(retrieved, filename)
