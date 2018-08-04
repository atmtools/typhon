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
from collections import OrderedDict
from os.path import join, dirname

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from typhon.collocations import collapse, Collocator
from typhon.utils.timeutils import to_datetime
import xarray as xr

from ..common import RetrievalProduct

__all__ = [
    'SPAREICE',
]

WEIGHTS_DIR = join(dirname(__file__), 'weights')


class SPAREICE:

    def __init__(self, file=None, collocator=None, processes=10,
                 verbose=2):
        self.verbose = verbose
        self.name = "SPARE-ICE"

        if collocator is None:
            self.collocator = Collocator(
                verbose=verbose,
            )
        else:
            self.collocator = collocator

        self.retrieval = RetrievalProduct(
            parameters_file=file, trainer=self._get_trainer(processes, verbose)
        )

    def _debug(self, msg):
        if self.verbose > 1:
            print(f"[{self.name}] {msg}")

    def _info(self, msg):
        if self.verbose > 0:
            print(f"[{self.name}] {msg}")

    def _get_estimator(self):
        """Return the default estimator"""

        # Estimators are normally objects that have a fit and predict method
        # (e.g. MLPRegressor from sklearn). To make their training easier we
        # scale the input data in advance. With Pipeline objects from sklearn
        # we can combine such steps easily since they behave like an
        # estimator object as well.
        estimator = MLPRegressor(max_iter=3500)

        return Pipeline([
            # SVM or NN work better if we have scaled the data in the first
            # place. MinMaxScaler is the simplest one. RobustScaler or
            # StandardScaler could be an alternative.
            ("scaler", RobustScaler(quantile_range=(15, 85))),
            # The "real" estimator:
            ("estimator", estimator),
        ])

    def _get_trainer(self, n_jobs, verbose):
        """Return the default trainer for the current estimator
        """

        # To optimize the results, we try different hyper parameters by
        # using a grid search
        hidden_layer_sizes = [
            (13, 10), (15, 10, 3), (12, 5),
        ]
        common = {
            'estimator__activation': ['relu', 'tanh'],
            'estimator__hidden_layer_sizes': hidden_layer_sizes,
            'estimator__random_state': [0, 5, 70, 100, 3452],
            #'alpha': 10.0 ** -np.arange(1, 7),
        }
        hyper_parameter = [
            {   # Hyper parameter for lbfgs solver
                'estimator__solver': ['lbfgs'],
                **common
            },
            # {  # Hyper parameter for adam solver
            #     'solver': ['adam'],
            #     'batch_size': [200, 1000],
            #     'beta_1': [0.95, 0.99],
            #     'beta_2': [0.95, 0.99],
            #     **common
            # },
        ]

        return GridSearchCV(
            self._get_estimator(), hyper_parameter, n_jobs=n_jobs,
            refit=True, cv=3, verbose=verbose,
        )

    def load_standard_weights(self):
        try:
            self.retrieval.load_parameters(
                join(WEIGHTS_DIR, "standard.json")
            )
        except Exception as e:
            print("Could not load the standard weights of SPARE-ICE!")
            raise e

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

    def get_inputs(self, data, fields=None):
        """Get the input fields for SPARE-ICE training / retrieval"""

        # Check whether the data is coming from a twice-collocated dataset:
        if "MHS_2C-ICE/MHS/scnpos" in data.variables:
            prefix = "MHS_2C-ICE/"
        else:
            prefix = ""

        inputs = pd.DataFrame(OrderedDict([
            ["mhs_channel1", data[f"{prefix}MHS/Data/btemps"].isel(
                **{f"{prefix}MHS/channel": 0}
            )],
            ["mhs_channel2", data[f"{prefix}MHS/Data/btemps"].isel(
                **{f"{prefix}MHS/channel": 1}
            )],
            ["mhs_channel3", data[f"{prefix}MHS/Data/btemps"].isel(
                **{f"{prefix}MHS/channel": 2}
            )],
            ["mhs_channel4", data[f"{prefix}MHS/Data/btemps"].isel(
                **{f"{prefix}MHS/channel": 3}
            )],
            ["mhs_channel5", data[f"{prefix}MHS/Data/btemps"].isel(
                **{f"{prefix}MHS/channel": 4}
            )],
            ["cloud_filter",
                 data["AVHRR/Data/btemps_mean"].isel(
                     **{"AVHRR/channel": 4}
                 ) - data["AVHRR/Data/btemps_mean"].isel(
                    **{"AVHRR/channel": 3})
            ],
            ["lat", data["lat"]],
            ["sea_mask", data["sea_mask"].astype(float)],
            ["mhs_scnpos", data[f"{prefix}MHS/scnpos"]],
            ["solar_azimuth_angle",
                data[f"{prefix}MHS/Geolocation/Solar_azimuth_angle"]],
            ["solar_zenith_angle",
                data[f"{prefix}MHS/Geolocation/Solar_zenith_angle"]],
            ["avhrr_channel3", data["AVHRR/Data/btemps_mean"].isel(
                **{"AVHRR/channel": 2}
            )],
            ["avhrr_channel4", data["AVHRR/Data/btemps_mean"].isel(
                **{"AVHRR/channel": 3}
            )],
            ["avhrr_channel4_std", data["AVHRR/Data/btemps_std"].isel(
                **{"AVHRR/channel": 3}
            )],
            ["avhrr_channel5", data["AVHRR/Data/btemps_mean"].isel(
                **{"AVHRR/channel": 4}
            )],
        ]))

        if fields is not None:
            self._info(f"Use only {fields}")
            inputs = inputs[fields]

        return inputs

    @staticmethod
    def get_targets(data):
        """Get the target fields for SPARE-ICE training"""
        targets = pd.DataFrame({
            "iwp_log10": np.log10(
                data["MHS_2C-ICE/2C-ICE/ice_water_path_mean"]
            )
        })
        return targets

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
            self.load_standard_weights()

        # We have to rename the variables when they come from collocations:
        if from_collocations and isinstance(data, xr.Dataset):
            inputs = self.get_inputs(data, self.retrieval.parameter["inputs"])
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
        if not as_log10 and retrieved is not None:
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
            self._info(
                f"Retrieve SPARE-ICE for {data.attrs['start_time']} to "
                f"{data.attrs['end_time']}"
            )
            # Remove NaNs from the data:
            data = data.dropna(dim="collocation")

            retrieved = self.retrieve(data, from_collocations=True)

            if retrieved is None:
                continue

            retrieved = retrieved.to_xarray()
            retrieved.rename({"index": "collocation"}, inplace=True)
            retrieved = retrieved.drop("collocation")

            # Add more information:
            retrieved["iwp"].attrs = {
                "units": "g/m^2",
                "name": "Ice Water Path",
                "description": "Ice Water Path retrieved by SPARE-ICE"
            }
            retrieved["lat"] = data["lat"]
            retrieved["lon"] = data["lon"]
            retrieved["time"] = data["time"]
            retrieved["scnpos"] = data["MHS/scnpos"]

            filename = output.get_filename(
                [to_datetime(data.attrs["start_time"]),
                 to_datetime(data.attrs["end_time"])], fill=attributes
            )

            # Write the data to the file.
            self._info(f"Store SPARE-ICE to \n{filename}")
            output.write(retrieved, filename)

    def score(self, data):
        # use the standard weights of SPARE-ICE:
        if not self.retrieval.is_trained():
            self.load_standard_weights()

        return self.retrieval.score(
            self.get_inputs(data),
            self.get_targets(data)
        )

    @staticmethod
    def split_data(data, test_ratio=None, shuffle=True):
        indices = np.arange(data.collocation.size)
        if shuffle:
            r = np.random.RandomState(1234)
            r.shuffle(indices)

        if test_ratio is None:
            test_ratio = 0.2

        boundary = int(data.collocation.size * (1 - test_ratio))

        # Make the training and testing data:
        return (
            data.isel(collocation=indices[:boundary]),
            data.isel(collocation=indices[boundary:]),
        )

    def train(self, data, fields=None):
        self._info("Train SPARE-ICE")
        train_score = self.retrieval.train(
            self.get_inputs(data, fields),
            self.get_targets(data),
        )
        self._info(f"Training score: {train_score:.2f}")