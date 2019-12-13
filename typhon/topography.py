"""
The topography module provides interfaces to global elevation models.
So far, only an interface to the `SRTM30
<https://dds.cr.usgs.gov/srtm/version2_1/SRTM30/srtm30_documentation.pdf>`_ data
set is provided, which has a resolution of 1 km.

Elevation data is downloaded on the fly but is cached to speed up subsequent
access. The interfaces uses the path pointed to by the :code:`TYPHON_DATA_PATH`
environment variable as data cache. This means that data is downloaded only
when they are not found in the cache.

.. note:: If :code:`TYPHON_DATA_PATH` is not set, the location of the file cache
    will be determined from the :code:`XDG_CACHE_HOME` environment variable and,
    if this is not defined, default to :`${HOME}/.typhon/topography`.

The module can be used in two ways: 
 1. by extracting the elevation data at native resolution
 2. by interpolating to elevation data at arbitrary locations

The two different use cases are described below.

Native resolution
-----------------

Extracting elevation data at native resolution for a given rectangular domain
 is done using the :code:`SRTM30.elevation` function. The function returns a
 tuple consisting of the latitude and longitude grids as well as the elevation
 data in meters.

.. code-block:: python

    lat_min = 50
    lon_min = 10
    lat_max = 60
    lon_max = 20
    lats, lons, z = SRTM30.elevation(lat_min, lon_min, lat_max, lon_max)

Interpolation to given coordinates
----------------------------------

Interpolation of the elevation data to arbitrary coordinates can be performed
using the :code:`interpolate` method. Interpolation uses nearest neighbor
interpolation and is implemented using a :code:`KDTree`.
Interpolating the SRTM30 data to given latitude and longitude grids can be done
as follows:

.. code-block:: python

    lat_min = 50
    lon_min = 10
    lat_max = 60
    lon_max = 20
    lats = np.linspace(lat_min, lat_max, 101)
    lons = np.linspace(lon_min, lon_max, 101)
    z = SRTM30.interpolate(lat, lont)
"""
import os
import shutil
import urllib
import zipfile

import numpy as np

import typhon
from typhon.environment import environ

_data_path = None

def _get_data_path():
    global _data_path
    if _data_path is None:
        if "TYPHON_DATA_PATH" in environ:
            _data_path = os.path.join(environ["TYPHON_DATA_PATH"], "topography")
        elif "XDG_CACHE_HOME" in environ:
            _data_path = environ["XDG_CACHE_HOME"]
        else:
            home = os.path.expandvars("~")
            _data_path = os.path.join(home, ".cache", "typhon", "topography")
        if not os.path.exists(_data_path):
            os.makedirs(_data_path)
    return _data_path

def _latlon_to_cart(lat, lon, R = typhon.constants.earth_radius):
    """
    Simple conversion of latitude and longitude to Cartesian coordinates.
    Approximates the Earth as sphere with radius :code:`R` and computes
    cartesian x, y, z coordinates with the center of the Earth as origin.

    Args:
        lat: Array of latitude coordinates.
        lon: Array of longitude coordinates.
        R: The radius to assume.
    Returns:
        Tuple :code:`(x, y, z)` of arrays :code:`x, y, z` containing the
        resulting x-, y- and z-coordinates.
    """
    lat = np.radians(lat)
    lon = np.radians(lon)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z

def _do_overlap(rect_1,
                rect_2):
    """
    Determines whether the two rectangles have overlap.

    Args:
        rect_1: Tuple :code:`(lat_min, lon_min, lat_max, lon_max) describing
                a rectangular tile.
        rect_2: Tuple :code:`(lat_min, lon_min, lat_max, lon_max) describing
                a rectangular tile.

    Returns:
        True if the two rectangles overlap.
    """
    lat_min_1, lon_min_1, lat_max_1, lon_max_1 = rect_1
    lat_min_2, lon_min_2, lat_max_2, lon_max_2 = rect_2
    lat_min = max(lat_min_1, lat_min_2)
    lon_min = max(lon_min_1, lon_min_2)
    lat_max = min(lat_max_1, lat_max_2)
    lon_max = min(lon_max_1, lon_max_2)
    return (lat_min < lat_max) and (lon_min < lon_max)

class SRTM30:
    """
    Interface to version 2.1 of SRTM30 digital elevation model.

    The data set has a resolution of about 1 km and covers all land masses
    except Antarctica.
    """
    _tile_height = 6000
    _tile_width = 4800
    _dlat = 50.0 / _tile_height
    _dlon = 40.0 / _tile_width

    _tiles = [("w180n90",  40, -180,  90, -140),
              ("w140n90",  40, -140,  90, -100),
              ("w100n90",  40, -100,  90,  -60),
              ("w060n90",  40,  -60,  90,  -20),
              ("w020n90",  40,  -20,  90,   20),
              ("e020n90",  40,   20,  90,   60),
              ("e060n90",  40,   60,  90,  100),
              ("e100n90",  40,  100,  90,  140),
              ("e140n90",  40,  140,  90,  180),
              ("w180n40", -10, -180,  40, -140),
              ("w140n40", -10, -140,  40, -100),
              ("w100n40", -10, -100,  40,  -60),
              ("w060n40", -10,  -60,  40,  -20),
              ("w020n40", -10,  -20,  40,   20),
              ("e020n40", -10,   20,  40,   60),
              ("e060n40", -10,   60,  40,  100),
              ("e100n40", -10,  100,  40,  140),
              ("e140n40", -10,  140,  40,  180),
              ("w180s10", -60, -180, -10, -140),
              ("w140s10", -60, -140, -10, -100),
              ("w100s10", -60, -100, -10,  -60),
              ("w060s10", -60,  -60, -10,  -20),
              ("w020s10", -60,  -20, -10,   20),
              ("e020s10", -60,   20, -10,   60),
              ("e060s10", -60,   60, -10,  100),
              ("e100s10", -60,  100, -10,  140),
              ("e140s10", -60,  140, -10,  180)]

    @staticmethod
    def get_tiles(lat_min, lon_min, lat_max, lon_max):
        """
        Get names of the tiles that contain the data of the given rectangular
        region of interest (ROI).

        Args:
            lat_min: The latitude of the lower left corner of the ROI
            lon_min: The longitude of the lower left corner of the ROI
            lat_max: The latitude of the upper right corner of the ROI
            lon_max: The longitude of the upper right corner of the ROI

        Return:

            List of tile names that contain the elevation data for the ROI.
        """
        lon_min = lon_min % 360
        if lon_min > 180:
            lon_min -= 360

        lon_max = lon_max % 360
        if lon_max > 180:
            lon_max -= 360


        fits = []
        for t in SRTM30._tiles:
            name, lat_min_1, lon_min_1, lat_max_1, lon_max_1 = t
            if _do_overlap((lat_min, lon_min, lat_max, lon_max),
                          (lat_min_1, lon_min_1, lat_max_1, lon_max_1)):
                fits += [name]
        return fits

    @staticmethod
    def get_bounds(name):
        """
        Get the bounds of tile with a given name.
        Args:
            name(str): The name of the tile.
        Returns:
            Tuple :code:(`lat_min`, `lon_min`, `lat_max`, `lon_max`) describing
            the bounding box of the tile with the given name.
        """
        tile = [t for t in SRTM30._tiles if t[0] == name][0]
        _, lat_min, lon_min, lat_max, lon_max = tile
        return lat_min, lon_min, lat_max, lon_max

    @staticmethod
    def get_grids(name):
        """
        Returns the latitude-longitude grid of the tile with the given name.

        Args:
            name(str): The name of the tile.
        Returns:
            Tuple :code:(`lat_grid`, `lon_grid`) containing the one dimensional
            latitude and longitude grids corresponding to the given tile.
        """
        lat_min, lon_min, lat_max, lon_max = SRTM30.get_bounds(name)
        start = lat_min + 0.5 * SRTM30._dlat
        stop = lat_max - 0.5 * SRTM30._dlat
        lat_grid = np.linspace(start, stop, SRTM30._tile_height)[::-1]

        start = lon_min + 0.5 * SRTM30._dlon
        stop = lon_max - 0.5 * SRTM30._dlon
        lon_grid = np.linspace(start, stop, SRTM30._tile_width)

        return lat_grid, lon_grid

    @staticmethod
    def get_native_grids(lat_min, lon_min, lat_max, lon_max):
        """
        Returns the latitude and longitude grid at native SRTM30 resolution
        that are included in the given rectangle.

        Args:
            lat_min: The latitude coordinate of the lower left corner.
            lon_min: The longitude coordinate of the lower left corner.
            lat_max: The latitude coordinate of the upper right corner.
            lon_max: The latitude coordinate of the upper right corner.
        Returns:
            Tuple :code:`(lats, lons)` of 1D-arrays containing the latitude
            and longitude coordinates of the SRTM30 data points within the
            given rectangle.
        """
        i = (90 - lat_max) / SRTM30._dlat
        i_max = np.trunc(i)
        if not i_max < i:
            i_max = i_max + 1
        i = (90 - lat_min) / SRTM30._dlat
        i_min = np.trunc(i)
        lat_grid = 90 + 0.5 * SRTM30._dlat - np.arange(i_max, i_min + 1) * SRTM30._dlat

        j = (lon_max + 180) / SRTM30._dlon
        j_max = np.trunc((lon_max + 180.0) / SRTM30._dlon)
        if not j_max < j:
            j_max = j_max - 1

        j_min = np.trunc((lon_min + 180.0) / SRTM30._dlon)
        lon_grid = -180 + 0.5 * SRTM30._dlon
        lon_grid += np.arange(j_min, j_max + 1) * SRTM30._dlon

        return lat_grid, lon_grid

    @staticmethod
    def download_tile(name):
        """
        This function will download and extract the tile with the given name.
        The data is stored in the path pointed to by the :code:`_data_path`
        attribute of the module.

        Args:
            name(str): The name of the tile to download.
        """
        base_url = "https://dds.cr.usgs.gov/srtm/version2_1/SRTM30"
        url = base_url + "/" + name + "/" + name + ".dem.zip"
        r = urllib.request.urlopen(url)

        filename = os.path.join(_get_data_path(), name + ".dem.zip")
        path = os.path.join(filename)
        with open(path, 'wb') as f:
            shutil.copyfileobj(r, f)

        # Extract zip file.
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(filename))

    @staticmethod
    def get_tile(name):
        """
        Get tile with the given name.

        Check the cache for the tile with the given name. If not found, the
        tile is download.

        Args:
            name(str): The name of the tile.
        """
        dem_file = os.path.join(_get_data_path(), (name + ".dem").upper())
        if not (os.path.exists(dem_file)):
            SRTM30.download_tile(name)
        y = np.fromfile(dem_file, dtype = np.dtype('>i2')).reshape(SRTM30._tile_height,
                                                                   SRTM30._tile_width)
        return y

    @staticmethod
    def get_tree(name):
        """
        Get KD-tree for the tile with the given name.

        Args:
            name(str): The name of the tile.
        """
        from pykdtree.kdtree import KDTree
        lat_grid, lon_grid = SRTM30.get_grids(name)
        lat_grid, lon_grid = np.meshgrid(lat_grid, lon_grid, indexing = "ij")
        x, y, z = _latlon_to_cart(lat_grid, lon_grid)
        X = np.concatenate([x.reshape(-1, 1, order = "C"),
                            y.reshape(-1, 1, order = "C"),
                            z.reshape(-1, 1, order = "C")], axis = 1)
        tree = KDTree(X.astype(np.float32))
        return tree

    @staticmethod
    def elevation(lat_min,
                  lon_min,
                  lat_max,
                  lon_max):
        """
        Return elevation data at native resolution in the a given rectangular
        domain.

        Args:
            lat_min(float): Latitude coordinate of the lower-left corner
            lon_min(float): Longitude coordinate of the lower-left corner.
            lat_max(float): Latitude coordinate of the upper-right corner
            lon_max(float): Longitude coordinate of the upper-right corner.

        """
        lats_d, lons_d = SRTM30.get_native_grids(lat_min,
                                                 lon_min,
                                                 lat_max,
                                                 lon_max)
        lat_min = lats_d.min() - 0.5 * SRTM30._dlat
        lat_max = lats_d.max() + 0.5 * SRTM30._dlat
        lon_min = lons_d.min() - 0.5 * SRTM30._dlon
        lon_max = lons_d.max() + 0.5 * SRTM30._dlon

        elevation = np.zeros(lats_d.shape + lons_d.shape)
        tiles = SRTM30.get_tiles(lat_min, lon_min, lat_max, lon_max)
        for t in tiles:
            dem = SRTM30.get_tile(t)
            lats, lons = SRTM30.get_grids(t)
            lat_min_s, lon_min_s, lat_max_s, lon_max_s = SRTM30.get_bounds(t)

            inds_lat = np.logical_and(lat_min <= lats, lats < lat_max)
            inds_lon = np.logical_and(lon_min <= lons, lons < lon_max)
            inds_s = np.logical_and(inds_lat.reshape(-1, 1),
                                    inds_lon.reshape(1, -1))

            inds_lat = np.logical_and(lat_min_s <= lats_d, lats_d < lat_max_s)
            inds_lon = np.logical_and(lon_min_s <= lons_d, lons_d < lon_max_s)
            inds_d = np.logical_and(inds_lat.reshape(-1, 1),
                                    inds_lon.reshape(1, -1))

            elevation[inds_d] = dem[inds_s]

        return lats_d, lons_d, elevation

    @staticmethod
    def interpolate(lats,
                    lons,
                    n_neighbors = 1):
        """
        Interpolate elevation data to the given coordinates.

        Uses KD-tree-based nearest-neighbor interpolation to interpolate
        the elevation data to arbitrary grids.

        Args:
            lats: Array containing latitude coordinates.
            lons: Array containing longitude coordinates.
            n_neighbors: Number of neighbors over which to average the elevation
                data.

        """
        lat_min = lats.min()
        lat_max = lats.max()
        lon_min = lons.min()
        lon_max = lons.max()
        tiles = SRTM30.get_tiles(lat_min, lon_min, lat_max, lon_max)

        elevation = np.zeros(lats.shape)
        for t in tiles:
            dem = SRTM30.get_tile(t).ravel()
            tree = SRTM30.get_tree(t)

            lat_min, lon_min, lat_max, lon_max = SRTM30.get_bounds(t)
            inds_lat = np.logical_and(lat_min <= lats, lats < lat_max)
            inds_lon = np.logical_and(lon_min <= lons, lons < lon_max)
            inds = np.logical_and(inds_lat, inds_lon)

            X = np.zeros((inds.sum(), 3))
            x, y, z = _latlon_to_cart(lats[inds], lons[inds])
            X[:, 0] = x
            X[:, 1] = y
            X[:, 2] = z

            _, neighbors = tree.query(np.asarray(X, np.float32), n_neighbors)
            if neighbors.size > 0:
                if len(neighbors.shape) > 1:
                    elevation[inds] = dem[neighbors].mean(axis = (1))
                else:
                    elevation[inds] = dem[neighbors]
        return elevation
