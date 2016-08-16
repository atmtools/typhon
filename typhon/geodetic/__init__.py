# -*- coding: utf-8 -*-

"""Functions for handling geographical coordinate systems
and reference ellipsoids.

Unless otherwise stated functions are ported from atmlab-2-3-181.

"""
import numpy as np
from numpy.lib import scimath

from typhon import constants


__all__ = [
    'ellipsoidmodels',
    'ellipsoid_r_geodetic',
    'ellipsoid_r_geocentric',
    'ellipsoid2d',
    'ellipsoidcurvradius',
    'sind',
    'cosd',
    'tand',
    'asind',
    'inrange',
    'cart2geocentric',
    'geocentric2cart',
    'cart2geodetic',
    'geodetic2cart',
    'geodetic2geocentric',
    'geocentric2geodetic',
]


def sind(x):
    """Sine of argument in degrees."""
    return np.sin(np.deg2rad(x))


def cosd(x):
    """Cosine of argument in degrees."""
    return np.cos(np.deg2rad(x))


def tand(x):
    """Tangent of argument in degrees."""
    return np.tan(np.deg2rad(x))


def asind(x):
    """Inverse sine in degrees."""
    return np.arcsin(np.deg2rad(x))


def inrange(x, minx, maxx, text=None):
    """Test if x is within given bounds.

    Parameters:
        x: Variable to test.
        minx: Lower boundary.
        maxx: Upper boundary.
        text (str): Addiitional warning text.

    Raises:
        Exception: If value is out of bounds.

    """
    if np.min(x) < minx or np.max(x) > maxx:
        if text is None:
            raise Exception('Range out of bound [{}, {}]'.format(minx, maxx))
        else:
            raise Exception(
                'Range out of bound [{}, {}]: {}'.format(minx, maxx, text)
                )


class ellipsoidmodels():
    """Provide data for different reference ellipsoids.

    The following models are covered:

        * SphericalEarth     (radius set as constants.earth_radius)
        * WGS84
        * SphericalVenus     (radius same as used in ARTS)
        * SphericalMars      (radius same as used in ARTS)
        * EllipsoidMars
        * SphericalJupiter   (radius same as used in ARTS)

    Examples:
        >>> e = ellipsoidmodels()

    """
    __credits__ = 'Patrick Eriksson'

    def __init__(self):
        self._data = {
            "SphericalEarth": (constants.earth_radius, 0),
            "WGS84": (6378137, 0.0818191908426),
            "SphericalVenus": (6051800.0, 0),
            "SphericalMars": (3389500.0, 0),
            "EllipsoidMars": (3396190.0, 0.1083),
            "SphericalJupiter": (69911000.0, 0),
            }

    def __getitem__(self, model):
        return self.get(model)

    def get(self, model='WGS84'):
        """Return data for different reference ellipsoids.

        Parameters:
            model (str): Model ellipsoid.

        Returns:
            tuple: Equatorial radius (r), eccentricity (e)

        Examples:
            >>> e['WGS84']
            (6378137, 0.0818191908426)
            >>> e.get('WGS84')
            (6378137, 0.0818191908426)
            >>> ellipsoidmodel()['WGS84']
            (6378137, 0.0818191908426)

        """
        if model in self._data:
            return self._data.__getitem__(model)
        else:
            raise Exception('Unknown ellipsoid model "{}".'.format(model))

    @property
    def models(self):
        """List of available models.

        Examples:
            >>> e.models
            ['SphericalVenus',
             'SphericalMars',
             'WGS84',
             'SphericalJupiter',
             'EllipsoidMars',
             'SphericalEarth']
        """
        return list(self._data.keys())


def ellipsoid_r_geocentric(ellipsoid, lat):
    """Geocentric radius of a reference ellipsoid.

    Gives the distance from the Earth's centre and the reference ellipsoid
    as a function of geoCENTRIC latitude.

    Note:
        To obtain the radii for *geodetic* latitude,
        use :func:`ellipsoid_r_geodetic`.

    Parameters:
        ellipsoid (tuple):  Model ellipsoid as returned
            by :class:`ellipsoidmodels`.
        lat: Geocentric latitudes.

    Returns:
        Radii.

    """
    __credits__ = 'Patrick Eriksson'

    if not (ellipsoid[1] >= 0 and ellipsoid[1] < 1):
        raise Exception('Invalid excentricity value in ellipsoid model.')

    if ellipsoid[1] == 0:
        r = np.ones(lat.shape) * ellipsoid[0]
    else:
        c = 1 - ellipsoid[1]**2
        b = ellipsoid[0] * np.sqrt(c)
        r = b / np.sqrt(c * cosd(lat)**2 + sind(lat)**2)

    return r


def ellipsoid_r_geodetic(ellipsoid, lat):
    """Geodetic radius of a reference ellipsoid.

    The calculation expressions are taken from radiigeo.pdf, found in the
    same folder as this function.

    Note:
        To obtain the radii for *geocentric* latitude,
        use :func:`ellipsoid_r_geocentric`.

    Parameters:
        ellipsoid (tuple):  Model ellipsoid as returned
            by :class:`ellipsoidmodels`.
        lat: Geodetic latitudes.

    Returns:
        Radii.

    """
    __credits__ = 'Patrick Eriksson'

    if not (ellipsoid[1] >= 0 and ellipsoid[1] < 1):
        raise Exception('Invalid excentricity value in ellipsoid model.')

    if ellipsoid[1] == 0:
        r = np.ones(lat.shape) * ellipsoid[0]
    else:
        e2 = ellipsoid[1]**2
        sin2 = sind(lat)**2
        r = (ellipsoid[0] * np.sqrt((1 - e2)**2 * sin2
             + cosd(lat) ** 2) / np.sqrt(1 - e2 * sin2)
             )
    return r


def ellipsoid2d(ellipsoid, orbitinc):
    """Approximate ellipsoid for 2D calculations.

    Determines an approximate reference ellipsoid following an orbit track.
    The new ellipsoid is determined simply, by determining the radius at the
    maximum latitude and from this value calculate a new eccentricity.
    The orbit is specified by giving the orbit inclination, that is
    normally a value around 100 deg for polar sun-synchronous orbits.

    Parameters:
        ellipsoid (tuple):  Model ellipsoid as returned
            by :class:`ellipsoidmodels`.
        orbitinc (float): Orbit inclination.

    Returns:
        tuple: Modified ellipsoid vector.

    """
    __credits__ = 'Patrick Erikkson'

    if not (ellipsoid[1] >= 0 and ellipsoid[1] < 1):
        raise Exception('Invalid excentricity value in ellipsoid model.')

    inrange(orbitinc, 0, 180, 'Invalid orbit inclination.')

    rp = ellipsoid_r_geocentric(ellipsoid, orbitinc)

    return ellipsoid[0], np.sqrt(1 - (rp / ellipsoid[0])**2)


def ellipsoidcurvradius(ellipsoid, lat_gd, azimuth):
    """Sets ellispoid to local curvature radius

    Calculates the curvature radius for the given latitude and azimuth
    angle, and uses this to set a spherical reference ellipsoid
    suitable for 1D calculations. The curvature radius is a better
    local approximation than using the local ellipsoid radius.

    The calculation expressions are taken from radiigeo.pdf, found in the
    same folder as this function.

    For exact result the *geodetic* latitude shall be used.

    Parameters:
    lat_gd: Geodetic latitude.
    azimuth: Azimuthal angle (angle from NS plane).
        If given curvature radii are returned, see above

    Returns:
        tuple: Modified ellipsoid.

    """
    __credits__ = 'Patrick Erikkson'

    if not (ellipsoid[1] >= 0 and ellipsoid[1] < 1):
        raise Exception('Invalid excentricity value in ellipsoid model.')

    aterm = 1 - ellipsoid[1]**2 * sind(lat_gd)**2
    rn = 1 / np.sqrt(aterm)
    rm = (1 - ellipsoid[1]**2) * (rn / aterm)
    e0 = (ellipsoid[0] /
          (cosd(azimuth)**2.0 / rm
           + sind(azimuth)**2.0 / rn)
          )
    e1 = 0

    return e0, e1


def cart2geocentric(x, y, z, lat0=None, lon0=None, za0=None, aa0=None):
    """Convert cartesian position to spherical coordinates.

    The geocentric Cartesian coordinate system is fixed with respect to the
    Earth, with its origin at the center of the ellipsoid and its X-, Y-,
    and Z-axes intersecting the surface at the following points:

        * X-axis: Equator at the Prime Meridian (0°, 0°)

        * Y-axis: Equator at 90-degrees East (0°, 90°)

        * Z-axis: North Pole (90°, 0°)

    A common synonym is Earth-Centered, Earth-Fixed coordinates, or ECEF.

    If the optional arguments are given, it is ensured that latitude and
    longitude are kept constant for zenith or nadir cases, and the longitude
    for N-S cases. The optional input shall be interpreted as the [x,y,z]
    is obtained by moving from [lat0,lon0] in the direction of [za0,aa0].

    Parameters:
        x: Coordinate in x dimension.
        y: Coordinate in y dimension.
        z: Coordinate in z dimension.
        lat0: Original latitude.
        lon0: Original longitude.
        za0: Orignal zenith angle.
        aa0: Orignal azimuth angle.

    Returns:
        tuple: Radius, Latitude, Longitude

    """
    __credits__ = 'Bengt Rydberg'

    r = np.sqrt(x**2 + y**2 + z**2)

    if not np.all(r > 0):
        raise Exception('This set of functions does not handle r > 0.')

    lat = np.rad2deg(np.arcsin(z / r))
    lon = np.rad2deg(np.arctan2(y, x))

    if all(x is not None for x in [lat0, lon0, za0, aa0]):
        for i in range(np.size(r)):
            if za0[i] < 1e-06 or za0[i] > 180 - 1e-06:
                lat[i] = lat0[i]
                lon[i] = lon0[i]

            if (abs(lat0[i]) < 90 - 1e-08 and
               (abs(aa0[i]) < 1e-06 or abs(aa0[i] - 180) < 1e-06)):
                if abs(lon[i] - lon0[i]) < 1:
                    lon[i] = lon0[i]
                else:
                    if lon0[i] > 0:
                        lon[i] = lon0[i] - 180
                    else:
                        lon[i] = lon0[i] + 180

    return r, lat, lon


def geocentric2cart(r, lat, lon):
    """Convert from spherical coordinate to a cartesian position.

     See :func:`cart2geocentric` for a defintion of the geocentric
     coordinate system.

     Parameters:
            r: Radius.
            lat: Latitude in degree.
            lon  Longitude in degree.

     Returns:
        tuple: Coordinate in x, y, z dimension.

    """
    __credits__ = 'Bengt Rydberg'

    if np.any(r == 0):
        error("This set of functions does not handle r = 0.")

    latrad = np.deg2rad(lat)
    lonrad = np.deg2rad(lon)

    x = r * np.cos(latrad)
    y = x * np.sin(lonrad)
    x = x * np.cos(lonrad)
    z = r * np.sin(latrad)

    return x, y, z


def cart2geodetic(x, y, z, ellipsoid=None):
    """Converts from cartesian to geodetic coordinates.

    The geodetic coordinates refer to the reference ellipsoid
    specified by input ellipsoid.
    See module docstring for a defintion of the geocentric coordinate system.

    Parameters:
        x: Coordinates in x dimension.
        y: Coordinates in y dimension.
        z: Coordinates in z dimension.
        ellipsoid: A tuple with the form (semimajor axis, eccentricity).
            Default is 'WGS84' from :class:`ellipsoidmodels`.

    Returns:
        tuple: Geodetic height, latitude and longitude

    """
    __credits__ = 'Bengt Rydberg'

    if ellipsoid is None:
        ellipsoid = ellipsoidmodels()['WGS84']

    if not (ellipsoid[1] >= 0 and ellipsoid[1] < 1):
        raise Exception('Invalid excentricity value in ellipsoid model.')

    lon = np.rad2deg(np.arctan2(y, x))
    B0 = np.arctan2(z, np.hypot(x, y))
    B = np.ones(B0.shape)
    e2 = ellipsoid[1]**2
    while (np.any(np.abs(B - B0) > 1e-10)):
        N = ellipsoid[0] / np.sqrt(1 - e2 * np.sin(B0)**2)
        h = np.hypot(x, y) / np.cos(B0) - N
        B = B0.copy()
        B0 = np.arctan(z/np.hypot(x, y) * ((1-e2*N/(N+h))**(-1)))

    lat = np.rad2deg(B)

    return h, lat, lon


def geodetic2cart(h, lat, lon, ellipsoid=None):
    """Converts from geodetic to geocentric cartesian coordinates.

    The geodetic coordinates refer to the reference ellipsoid
    specified by input ellipsoid.
    See module docstring for a defintion of the geocentric coordinate system.

    Parameters:
        h: Geodetic height (height above the reference ellipsoid).
        lat: Geodetic latitude.
        lon: Geodetic longitude.
        ellipsoid: A tuple with the form (semimajor axis, eccentricity).
            Default is 'WGS84' from :class:`ellipsoidmodels`.

    Returns:
        tuple: x, y, z coordinates.

    """
    __credits__ = 'Bengt Rydberg'

    if ellipsoid is None:
        ellipsoid = ellipsoidmodels()['WGS84']

    if not (ellipsoid[1] >= 0 and ellipsoid[1] < 1):
        raise Exception('Invalid excentricity value in ellipsoid model.')

    a = ellipsoid[0]
    e2 = ellipsoid[1] ** 2

    N = a / np.sqrt(1 - e2 * sind(lat)**2)
    x = (N + h) * (cosd(lat)) * (cosd(lon))
    y = (N + h) * (cosd(lat)) * (sind(lon))
    z = (N * (1 - e2) + h) * (sind(lat))

    return x, y, z


def geodetic2geocentric(h, lat, lon, ellipsoid=None, **kwargs):
    """Converts from geodetic to geocentric coordinates.

    The geodetic coordinates refer to the reference ellipsoid
    specified by input ellipsoid.
    See module docstring for a defintion of the geocentric coordinate system.

    Parameters:
        h: Geodetic height (height above the reference ellipsoid).
        lat: Geodetic latitude.
        lon: Geodetic longitude.
        kwargs: Additional keyword arguments for :func:`cart2geocentric`.
        ellipsoid: A tuple with the form (semimajor axis, eccentricity).
            Default is 'WGS84' from :class:`ellipsoidmodels`.

    Returns:
        tuple: Radius, geocentric latiude, geocentric longitude

    """
    __credits__ = 'Bengt Rydberg'

    if ellipsoid is None:
        ellipsoid = ellipsoidmodels()['WGS84']

    if not (ellipsoid[1] >= 0 and ellipsoid[1] < 1):
        raise Exception('Invalid excentricity value in ellipsoid model.')

    cart = geodetic2cart(h, lat, lon, ellipsoid)
    return cart2geocentric(*cart, **kwargs)


def geocentric2geodetic(r, lat, lon, ellipsoid=None):
    """Converts from geocentric to geodetic coordinates.

    The geodetic coordinates refer to the reference ellipsoid
    specified by input ellipsoid.
    See module docstring for a defintion of the geocentric coordinate system.

    Returns:
        tuple: Geodetic height, latitude and longitude

    Parameters:
        r: Radius:
        lat: Geocentric latitude.
        lon: Geocentric longitude.
        ellipsoid: A tuple with the form (semimajor axis, eccentricity).
            Default is 'WGS84' from :class:`ellipsoidmodels`.

    """
    __credits__ = 'Bengt Rydberg'

    if ellipsoid is None:
        ellipsoid = ellipsoidmodels()['WGS84']

    if not (ellipsoid[1] >= 0 and ellipsoid[1] < 1):
        raise Exception('Invalid excentricity value in ellipsoid model.')

    cart = geocentric2cart(r, lat, lon)
    return cart2geodetic(*cart, ellipsoid)
