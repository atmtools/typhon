"""
This is not a real test file but rather an example draft that shows how it should look like to use SPARE-ICE.
"""

import typhon.collocations
import typhon.datasets
import typhon.spareice

## COLLOCATIONS

# Define the two datasets that you want to collocate
dataset1 = typhon.datasets.Dataset(
    files="/path2/to/files/{year}/{month}/{day}.nc.gz",
    file_handler=typhon.datasets.handler.MHS,
    name="MHS",
)

dataset2 = typhon.datasets.Dataset(
    files="/path3/to/files/{year}/{month}/{day}.hdf.zip",
    file_handler=typhon.datasets.handler.HDF,
    name="CloudSat",
)

# Define the collocated dataset
collocated_data = typhon.collocations.Dataset(
    files="/path/to/files/{YEAR}/{MONTH}/{DAY}.nc.gz",
    file_handler=typhon.datasets.handler.NetCDF, # Actually, handler.NetCDF should be the default file handler for all datasets.
    datasets=(dataset1, dataset2),
    fields=(
        ("bt_1", "bt_2", "bt_3", "bt_4", "bt_5",),
        ("IWP_RO", "mean_IWP")
    ),
)

# HDFHandler and NetCDFHandler inherit from typhon.datasets.FileHandler which provides the abstract methods read() and
# write(). They should be implemented in each subclass.
# typhon.datasets.FileHandler.read() should read a file and return a numpy.array (or xarray).
# typhon.datasets.FileHandler.write() should write a numpy.array (or xarray) to a file.
# Then, the Dataset classes do not have to deal with specific file formats any longer.

# Collocate those two datasets (the newly created collocation files will be stored as defined in the files parameter of the
# CollocatedDataset initialization from above).
collocated_data.collocate(
    start_date=(2016, 10, 1), end_date=(2016, 12, 1),
    temporal_limit=300, spatial_limit=10,
)

## SPARE-ICE - Training
spareice_trainer = typhon.spareice.Trainer(
    dataset=collocated_data,
    dataset_subdivision=(0.6, 0.2, 0.2), # fraction of training, validation and testing datasets
    inputs=(("MHS.bt", (2, 2, 3, 5, (5, 5, 6))), "MHS.bt[2][2]", "MHS.bt[3]", "MHS.bt_4", "MHS.bt_5"),
    outputs=("SPAREICE.IWP", "SPAREICE.cloud_probability"),
    targets=("CloudSat.IWP_RO", "CloudSat.mean_IWP"),
)

results = spareice_trainer.train(
    episodes=2000,
    batch_training=True,
)

spareice_trainer.save_parameters("/path/to/spareice/neural_nets/foo.json")

## SPARE-ICE - Retrieving

dataset_to_retrieve_from = typhon.datasets.Dataset(
    files="/path3/to/files/{YEAR}/{MONTH}/{DAY}.nc.gz",
    name="MHS"
)

spareice_retriever = typhon.spareice.Retriever(
    dataset=dataset_to_retrieve_from,
    parameter_file="/path/to/spareice/neural_nets/foo.json"
)

spareice_retriever.retrieve(
    start_date=(2015, 2, 1), end_date=(2015, 5, 1),
    output_path="/path/to/spareice/files/{YEAR}/{MONTH}/{DAY}.nc.gz"
)