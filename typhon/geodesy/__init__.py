# -*- coding: utf-8 -*-

"""Functions for handling geographical coordinate systems
and reference ellipsoids.

Unless otherwise stated functions are ported from atmlab-2-3-181.

"""
import numpy as np

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
    'great_circle_distance',
    'geographic_mean',
    'cartposlos2geocentric',
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


def inrange(x, minx, maxx, exclude='none', text=None):
    """Test if x is within given bounds.

    Parameters:
        x: Variable to test.
        minx: Lower boundary.
        maxx: Upper boundary.
        exclude (str): Exclude boundaries. Possible values are:
            'none', 'lower', 'upper' and 'both'
        text (str): Addiitional warning text.

    Raises:
        Exception: If value is out of bounds.

    """
    compare = {'none': (np.greater_equal, np.less_equal),
               'lower': (np.greater, np.less_equal),
               'upper': (np.greater_equal, np.less),
               'both': (np.greater, np.less),
               }

    greater, less = compare[exclude]

    if less(x, minx) or greater(x, maxx):
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

    .. Ported from atmlab. Original author: Patrick Eriksson
    """

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
    as a function of geo\ **centric** latitude.

    Note:
        To obtain the radii for **geodetic** latitude,
        use :func:`ellipsoid_r_geodetic`.

    Parameters:
        ellipsoid (tuple):  Model ellipsoid as returned
            by :class:`ellipsoidmodels`.
        lat (float or ndarray): Geocentric latitudes.

    Returns:
        float or ndarray: Radii.

    .. Ported from atmlab. Original author: Patrick Eriksson
    """
    errtext = 'Invalid excentricity value in ellipsoid model.'
    inrange(ellipsoid[1], 0, 1, exclude='upper', text=errtext)

    if ellipsoid[1] == 0:
        r = np.ones(np.shape(lat)) * ellipsoid[0]
    else:
        c = 1 - ellipsoid[1]**2
        b = ellipsoid[0] * np.sqrt(c)
        r = b / np.sqrt(c * cosd(lat)**2 + sind(lat)**2)

    return r


def ellipsoid_r_geodetic(ellipsoid, lat):
    """Geodetic radius of a reference ellipsoid.

    Gives the distance from the Earth's centre and the reference ellipsoid
    as a function of geo\ **detic** latitude.

    Note:
        To obtain the radii for **geocentric** latitude,
        use :func:`ellipsoid_r_geocentric`.

    Parameters:
        ellipsoid (tuple):  Model ellipsoid as returned
            by :class:`ellipsoidmodels`.
        lat (float or ndarray): Geodetic latitudes.

    Returns:
        float or ndarray: Radii.

    .. Ported from atmlab. Original author: Patrick Eriksson
    """
    errtext = 'Invalid excentricity value in ellipsoid model.'
    inrange(ellipsoid[1], 0, 1, exclude='upper', text=errtext)

    if ellipsoid[1] == 0:
        r = np.ones(np.shape(lat)) * ellipsoid[0]
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

    .. Ported from atmlab. Original author: Patrick Eriksson
    """
    errtext = 'Invalid excentricity value in ellipsoid model.'
    inrange(ellipsoid[1], 0, 1, exclude='upper', text=errtext)

    inrange(orbitinc, 0, 180,
            exclude='both',
            text='Invalid orbit inclination.')

    rp = ellipsoid_r_geocentric(ellipsoid, orbitinc)

    return ellipsoid[0], np.sqrt(1 - (rp / ellipsoid[0])**2)


def ellipsoidcurvradius(ellipsoid, lat_gd, azimuth):
    """Sets ellispoid to local curvature radius

    Calculates the curvature radius for the given latitude and azimuth
    angle, and uses this to set a spherical reference ellipsoid
    suitable for 1D calculations. The curvature radius is a better
    local approximation than using the local ellipsoid radius.

    For exact result the *geodetic* latitude shall be used.

    Parameters:
        lat_gd: Geodetic latitude.
        azimuth: Azimuthal angle (angle from NS plane).
            If given curvature radii are returned, see above.

    Returns:
        tuple: Modified ellipsoid.

    .. Ported from atmlab. Original author: Patrick Eriksson
    """
    errtext = 'Invalid excentricity value in ellipsoid model.'
    inrange(ellipsoid[1], 0, 1, exclude='upper', text=errtext)

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

    .. Ported from atmlab. Original author: Bengt Rydberg
    """
    r = np.sqrt(x**2 + y**2 + z**2)

    if np.any(r == 0):
        raise Exception("This set of functions does not handle r = 0.")

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

    .. Ported from atmlab. Original author: Bengt Rydberg
    """
    if np.any(r == 0):
        raise Exception("This set of functions does not handle r = 0.")

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

    .. Ported from atmlab. Original author: Bengt Rydberg
    """
    if ellipsoid is None:
        ellipsoid = ellipsoidmodels()['WGS84']

    errtext = 'Invalid excentricity value in ellipsoid model.'
    inrange(ellipsoid[1], 0, 1, exclude='upper', text=errtext)

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

    .. Ported from atmlab. Original author: Bengt Rydberg
    """
    if ellipsoid is None:
        ellipsoid = ellipsoidmodels()['WGS84']

    errtext = 'Invalid excentricity value in ellipsoid model.'
    inrange(ellipsoid[1], 0, 1, exclude='upper', text=errtext)

    a = ellipsoid[0]
    e2 = ellipsoid[1] ** 2

    N = a / np.sqrt(1 - e2 * sind(lat)**2)
    x = (N + h) * (cosd(lat)) * (cosd(lon))
    y = (N + h) * (cosd(lat)) * (sind(lon))
    # np.ones(np.shape(lon)): Ensure equal shape of x, y, z.
    z = (N * (1 - e2) + h) * (sind(lat)) * np.ones(np.shape(lon))

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

    .. Ported from atmlab. Original author: Bengt Rydberg
    """
    if ellipsoid is None:
        ellipsoid = ellipsoidmodels()['WGS84']

    errtext = 'Invalid excentricity value in ellipsoid model.'
    inrange(ellipsoid[1], 0, 1, exclude='upper', text=errtext)

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

    .. Ported from atmlab. Original author: Bengt Rydberg
    """
    if ellipsoid is None:
        ellipsoid = ellipsoidmodels()['WGS84']

    errtext = 'Invalid excentricity value in ellipsoid model.'
    inrange(ellipsoid[1], 0, 1, exclude='upper', text=errtext)

    cart = geocentric2cart(r, lat, lon)
    return cart2geodetic(*cart, ellipsoid)


def great_circle_distance(lat1, lon1, lat2, lon2, r=None):
    """Calculate the distance between two geograpgical positions.

    "As-the-crow-flies" distance between two points, specified by their
    latitude and longitude.

    If the optional argument *r* is given, the distance in m is returned.
    Otherwise the angular distance in degrees is returned.

    Parameters:
        lat1: Latitude of position 1.
        lon1: Longitude of position 1.
        lat2: Latitude of position 2.
        lon2: Longitude of position 2.
        r (float): The radius (common for both points).

    Returns:
        Distance, either in degress or m.

    .. Ported from atmlab. Original author: Patrick Eriksson
    """
    a = (sind((lat2 - lat1) / 2)**2
         + cosd(lat1) * (cosd(lat2))
         * (sind((lon2 - lon1) / 2)**2)
         )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    if r is None:
        return np.rad2deg(c)
    else:
        return r * c


def geographic_mean(lat, lon, h=0, ellipsoid=None):
    """Calculate mean position for set of coordinates.

    Parameters:
        lat (float or ndarray): Latitudes in degrees.
        lon (float or ndarray): Longitudes in degrees.
        h (float or ndarray): Optiional altitude for
            each coordinate (default is).
        ellipsoid: A tuple with the form (semimajor axis, eccentricity).
            Default is 'WGS84' from :class:`ellipsoidmodels`.

    Returns:
       tuple: Mean latitudes and longitudes in degrees.

    """
    if ellipsoid is None:
        ellipsoid = ellipsoidmodels()['WGS84']

    x, y, z = geodetic2cart(
        h,
        lat,
        lon,
        ellipsoid=ellipsoid)

    mh, mlat, mlon = cart2geodetic(
        np.mean(x),
        np.mean(y),
        np.mean(z),
        ellipsoid=ellipsoid)

    return mlat, mlon


def cartposlos2geocentric(x, y, z, dx, dy, dz, ppc=None,
                          lat0=None, lon0=None, za0=None, aa0=None):
    """Converts cartesian POS/LOS to spherical coordinates.

    Position is given as (x,y,z), while line-of-sight is given as (dx,dy,dz).
    The corresponding quantities in polar coordinates are (r,lat,lon)
    and (za,aa), respectively.

    See *Contents* for defintion of coordinate systems.

    If the optional arguments are given, it is ensured that latitude and
    longitude are kept constant for zenith or nadir cases, and the longitude
    and azimuth angle for N-S cases. The optional input shall be interpreted
    as the [x,y,z] is obtained by moving from [r0,lat0,lon0] in the direction
    of [za0,aa0].

    This version is different from the atmlab version by normalizing the los-
    vector and demanding all or nothing for the optional arguments to work.

    Parameters:
        x: Coordinate in x dimension.
        y: Coordinate in y dimension.
        z: Coordinate in z dimension.
        dx: LOS component in x dimension.
        dy: LOS component in y dimension.
        dz: LOS component in z dimension.
        ppc: Propagation path constant = r0*sin(za0).
        lat0: Original latitude.
        lon0: Original longitude.
        za0: Orignal zenith angle.
        aa0: Orignal azimuth angle.

    Returns:
        tuple(ndarray): Radius, Latitude, Longitude,
            Zenith angle, Azimuth angle

    .. Ported from atmlab. Original author: Bengt Rydberg

    """
    # Here be dragons!

    # Broadcast all input variables to the same shape.
    if (ppc is not None and za0 is not None and lat0 is not None
        and aa0 is not None and lon0 is not None):
        x, y, z, dx, dy, dz, ppc, lat0, lon0, za0, aa0 = np.broadcast_arrays(
            x, y, z, dx, dy, dz, ppc, lat0, lon0, za0, aa0)
    elif ppc is not None:
        x, y, z, dx, dy, dz, ppc = np.broadcast_arrays(
            x, y, z, dx, dy, dz, ppc)
    else:
        x, y, z, dx, dy, dz = np.broadcast_arrays(x, y, z, dx, dy, dz)

    r, lat, lon = cart2geocentric(x, y, z, lat0, lon0, za0, aa0)

    # Renormalize for length of the variables (not in atmlab)
    norm_r = np.sqrt(dx**2 + dy**2 + dz**2)
    dx = dx / norm_r
    dy = dy / norm_r
    dz = dz / norm_r

    coslat = np.cos(np.deg2rad(lat))
    sinlat = np.sin(np.deg2rad(lat))
    coslon = np.cos(np.deg2rad(lon))
    sinlon = np.sin(np.deg2rad(lon))
    dr = coslat * coslon * dx + sinlat * dz + coslat * sinlon * dy

    # Get LOS angle
    if ppc is None:
        za = np.rad2deg(np.arccos(dr))
    else:
        za = np.rad2deg(np.arcsin(ppc / r))
    aa = np.zeros(za.shape)

    # Fix zenith and azimuth angle with optional input only when all exists
    if za0 is not None and lat0 is not None and aa0 is not None and lon0 is not None:

        # Determine the type for zenith
        noz = np.logical_or(za0 < 1e-06, za0 > 180 - 1e-06)
        nan = np.isnan(za)
        pre = np.logical_and(~noz, nan)

        # Either set or do not
        za[noz] = za0[noz]
        za[pre] = 90.
        # NB: removed check for dr<0 since by putting dr==1 is more sensible

        # Determine the type for azimuth
        cir1 = abs(aa0) < 1e-06
        cir2 = np.logical_or(cir1, abs(aa0 - 180) < 1e-06)
        same = np.equal(lon, lon0)
        circ = np.logical_and(cir2, same)
        left = np.logical_and(cir1, ~same)
        right = np.logical_and(~cir1, ~same)

        # This should set all cases
        aa[circ] = aa0[circ]
        aa[left] = 180.
        aa[right] = 0.
    else:

        # Determine the type of calculations to be carried out
        noz = np.logical_or(za < 1e-06, za > 180 - 1e-06)
        pol = abs(lat) > 90 - 1e-08
        pre = np.logical_and(~noz, pol)
        non = np.logical_and(~noz, ~pol)
        aa[noz] = 0.
        aa[pre] = np.rad2deg(np.arctan2(dy[pre], dx[pre]))

        dlat = (- sinlat[non] * coslon[non] / r[non] * dx[non] + coslat[non] /
                r[non] * dz[non] - sinlat[non] * sinlon[non] / r[non] * dy[non]
                )
        dlon = (- sinlon[non] / coslat[non] / r[non] * dx[non] + coslon[non] /
                coslat[non] / r[non] * dy[non])
        aa[non] = (np.rad2deg(np.arccos(r[non] *
                   dlat / np.sin(np.deg2rad(za[non])))))

        fix = np.logical_or(np.isnan(aa), ~np.isreal(aa))
        aa[np.logical_and(fix, dlat >= 0)] = 0
        aa[np.logical_and(fix, dlat < 0)] = 180

        aa[np.logical_and(~fix, dlon < 0)] *= -1

    return r, lat, lon, za, aa
