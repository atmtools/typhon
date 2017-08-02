import datetime
import time

import numpy as np
import scipy.spatial
#import typhon.datasets
import typhon.geodesy
import typhon.spareice.datasets
import xarray as xr

__all__ = [
    "Dataset"
    ]


class Dataset(typhon.spareice.datasets.Dataset):
    """ Class that can find collocations amongst other datasets and store them. 
    """

    def __init__(self, datasets, **kwargs):
        """ 
        
        Args:
            datasets:
            **kwargs: 
        """
        super().__init__(**kwargs)

        # Check arguments:
        if len(datasets) != 2:
            raise ValueError("Need two datasets to perform collocations!")

        # Initialize member variables:
        self.datasets = datasets
        #self.fields = fields

    def collocate(self, start, end, fields, max_interval=300, max_distance=10, collapser=np.mean):
        """
        
        Args:
            start: 
            end:
            fields: 
            max_interval: 
            max_distance: 

        Returns:

        """

        max_interval = datetime.timedelta(seconds=max_interval)

        # Find all files from the primary dataset in the given period
        primaries = sorted(list(self.datasets[0].find_files(start, end)), key=lambda x: x[1])

        # All additional fields given by the user:
        primary_fields = list(set(fields[0]) | {"time", "lat", "lon"})
        secondary_fields = list(set(fields[1]) | {"time", "lat", "lon"})

        # The data of secondary files can overlap multiple primary files.
        secondary_data = None

        total_primaries_points, total_secondaries_points = 0, 0

        start_time = time.time()

        # Find all files from the primary dataset in the given period and search for their collocations.
        for index, primary in enumerate(primaries):
            print("Primary:", primary[0])

            # The starting time of the primary file is provided by .find_files(..). Since we assume a continuous
            # dataset, we can use the starting time of the next primary file as ending time for the first. For the last
            # primary file, we use the date which is given by the "end" parameter as ending time.
            # TODO: This approach misses the secondaries which start before the primary's starting time but still
            # overlap the time range. They should be found, too.
            secondaries_start = primary[1] - max_interval
            secondaries_end = primaries[index + 1][1] + max_interval if index != len(primaries) - 1 else end

            secondaries = sorted(self.datasets[1].find_files(secondaries_start, secondaries_end), key=lambda x: x[1])

            # Skip this primary file if there are no secondaries found.
            if not secondaries:
                continue

            # Read the data of the primary file
            primary_data = self.datasets[0].read(primary[0], primary_fields)

            print(primary_data)

            # We try to find collocations by building a 4-d tree (see https://en.wikipedia.org/wiki/K-d_tree) and
            # searching for the nearest neighbours. A k-d tree cannot handle latitude/longitude data, we have to convert
            # them to 3D-cartesian coordinates.
            x, y, z = typhon.geodesy.geocentric2cart(
                typhon.constants.earth_radius / 10e3, np.asarray(primary_data["lat"]), np.asarray(primary_data["lon"])
            )

            # The fourth dimension in the tree is the time. To avoid defining complicated distance metrics, we simply
            # scale the temporal dimension (seconds) into a spatial dimension (kilometers). Firstly, we have to
            # convert them from datetime objects to timestamps (seconds since 1970-1-1).
            t = (np.asarray(primary_data["time"].astype('uint64')) / 1e6) * max_distance / max_interval.total_seconds()

            primary_points = list(zip(x, y, z, t))
            #print(primary_points)
            primary_tree = scipy.spatial.cKDTree(primary_points, leafsize=x.shape[0])

            # Find all secondary files in this time range.
            for secondary in secondaries:
                print("Secondary:", secondary[0])
                secondary_data = self.datasets[1].read(secondary[0], secondary_fields)

                print(secondary_data)

                # We need to convert the secondary data as well:
                x, y, z = typhon.geodesy.geocentric2cart(
                    typhon.constants.earth_radius / 10e3, np.asarray(secondary_data["lat"]), np.asarray(secondary_data["lon"])
                )
                t = (np.asarray(secondary_data["time"].astype('uint64')) / 1e6) * max_distance / max_interval.total_seconds()

                secondary_points = list(zip(x, y, z, t))
                secondary_tree = scipy.spatial.cKDTree(secondary_points, leafsize=x.shape[0])

                # Search for all collocations:
                results = primary_tree.query_ball_tree(secondary_tree, max_distance)

                primary_indices = [i for i, found in enumerate(results) if found]

                # No collocations were found.
                if not primary_indices:
                    print("\tNo collocations found.")
                    continue

                secondary_indices = [found for found in results if found]

                print("\tFound {0} secondary data points to {1} primary data points.".format(
                    len(primary_indices), len(secondary_indices)))

                # Prepare the data array to write it to a file.
                collocation_data = xr.Dataset()
                collocation_data.attrs["files"] = [primary[0], secondary[0]]
                collocation_data.attrs["datasets"] = [self.datasets[0].name, self.datasets[1].name]

                # To save variables in netCDF format, we need to specify variable length
                encoding = {}

                # Add the indices
                collocation_data[self.datasets[0].name + ".indices"] = primary_indices
                collocation_data[self.datasets[1].name + ".indices"] = secondary_indices
                #encoding[self.datasets[1].name + ".indices"] = {"dtype" : }
                #print(secondary_indices[5:20])

                # Add the selected data from the primary dataset
                for variable in primary_data.keys():
                    collocation_data[self.datasets[0].name + "." + variable] = primary_data[variable][primary_indices]

                # Add the selected data from the secondary dataset
                for variable in secondary_data.keys():
                    data = [collapser(secondary_data[variable][indices]) for indices in secondary_indices]
                    collocation_data[self.datasets[1].name + "." + variable] = data

                filename = self.generate_filename_from_timestamp(
                    self.files,
                    datetime.datetime.utcfromtimestamp(primary_data["time"][primary_indices[0]].astype('O')/1e9))

                # Write the data to the file.
                print("\tWrite collocations to '{0}'".format(filename))
                self.write(filename, collocation_data)

                total_primaries_points += len(primary_indices)
                total_secondaries_points += len(secondary_indices)

        print("Needed {0:.2f} seconds to find {1} ({2}) and {3} ({4}) collocation points.".format(
            time.time()-start_time, total_primaries_points,
            self.datasets[0].name, total_secondaries_points, self.datasets[1].name))



    def _find_collocations(self, start, end):
        ...