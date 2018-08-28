from satpy import Scene
import xarray as xr

from .common import expects_file_info, FileHandler


class SatPy(FileHandler):
    """Wrapper for SatPy readers

    SatPy provides many readers for different satellite files. With this
    simple wrapper class, you can use them with typhon's FileSet without
    requiring any extra code!

    Warnings:
        At the moment, this wrapper works for retrieving fields with the
        same resolution only. Latitudes and longitudes are not retrieved.
    """

    def __init__(self, satpy_reader):
        super(SatPy, self).__init__()

        self.satpy_reader = satpy_reader

    @expects_file_info()
    def read(self, filename, fields=None, **kwargs):
        scene = Scene(
            reader=self.satpy_reader,
            filenames=[filename.path]
        )

        # If the user has not passed any fields to us, we load all per default.
        if fields is None:
            fields = scene.available_dataset_ids()

        # Load all selected fields
        scene.load(fields, **kwargs)

        if isinstance(fields[0], str):
            data_arrays = {field: scene.get(field) for field in fields}
        else:
            data_arrays = {field.name: scene.get(field) for field in fields}

        for name, array in data_arrays.items():
            array.name = name

        dataset = xr.merge(data_arrays.values())

        return dataset


