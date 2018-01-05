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
    'geocentricposlos2cart',
    'sphere_plane_intersection',
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
        r = (ellipsoid[0] * np.sqrt((1 - e2)**2 * sin2 +
             cosd(lat) ** 2) / np.sqrt(1 - e2 * sin2))
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
    e0 = (ellipsoid[0] / (cosd(azimuth)**2.0 / rm + sind(azimuth)**2.0 / rn))
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
    """Convert from cartesian to geodetic coordinates.

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
    if e2 == 0.0:
        h, lat, lon = cart2geocentric(x, y, z)
        h -= ellipsoid[0]
    else:
        while (np.any(np.abs(B - B0) > 1e-10)):
            N = ellipsoid[0] / np.sqrt(1 - e2 * np.sin(B0)**2)
            h = np.hypot(x, y) / np.cos(B0) - N
            B = B0.copy()
            B0 = np.arctan(z/np.hypot(x, y) * ((1-e2*N/(N+h))**(-1)))

        lat = np.rad2deg(B)

    return h, lat, lon


def geodetic2cart(h, lat, lon, ellipsoid=None):
    """Convert from geodetic to geocentric cartesian coordinates.

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
    """Convert from geodetic to geocentric coordinates.

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
    """Convert from geocentric to geodetic coordinates.

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
    a = (sind((lat2 - lat1) / 2)**2 + cosd(lat1) * (cosd(lat2)) *
         (sind((lon2 - lon1) / 2)**2))

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
    """Convert cartesian POS/LOS to spherical coordinates.

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

    # Broadcast all input variables to the same shape.  Atleast (1)
    if(ppc is not None and za0 is not None and lat0 is not None and
       aa0 is not None and lon0 is not None):
        x, y, z, dx, dy, dz, ppc, lat0, lon0, za0, aa0 = _broadcast(
            x, y, z, dx, dy, dz, ppc, lat0, lon0, za0, aa0)
    elif ppc is not None:
        x, y, z, dx, dy, dz, ppc = _broadcast(x, y, z, dx, dy, dz, ppc)
    else:
        x, y, z, dx, dy, dz = _broadcast(x, y, z, dx, dy, dz)

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
    if(za0 is not None and lat0 is not None and
       aa0 is not None and lon0 is not None):

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


def geocentricposlos2cart(r, lat, lon, za, aa):

    """
     Convert from spherical POS/LOS to cartesian POS/LOS

     See Contents for a defintion of the geocentric coordinate system.
     The local LOS angles are defined following the EAST-NORTH-UP system:

             za    aa

             90    0   points towards north

             90    90  points towards east

             0     aa  points up

     FORMAT  [x,y,z,dx,dy,dz]=geocentricposlos2cart(r,lat,lon,za,az)

     OUT
            x    Coordinate in x dimension

            y    Coordinate in y dimension

            z    Coordinate in z dimension

            dx   LOS component in x dimension

            dy   LOS component in y dimension

            dz   LOS

     IN
            r    Radius

            lat  Latitude

            lon  Longitude

            za   zenith angle

            aa   azimuth angle

     Ported from atmlab.  Original author: Bengt Rydberg 2011-10-31
    """

    r, lat, lon, za, aa = _broadcast(r, lat, lon, za, aa)

    if any(r == 0):
        raise Exception("This function is not handling the case of r = 0.")
    if any(lat < -90) or any(lat > 90):
            raise RuntimeError("The latitude is out of range")
    if any(lon < -180) or any(lon > 180):
            raise RuntimeError("The longitude is out of range")
    if any(za < 0) or any(za > 180):
            raise RuntimeError("The zenith angle is out of range")

    deg2rad = np.deg2rad(1)

    x = np.empty(r.shape)
    y = np.empty(r.shape)
    z = np.empty(r.shape)
    dx = np.empty(r.shape)
    dy = np.empty(r.shape)
    dz = np.empty(r.shape)

    at_pole = abs(lat) > (90 - 1e-8)

    if any(at_pole):
        s = np.sign(lat[at_pole])
        x[at_pole] = 0.
        y[at_pole] = 0.
        z[at_pole] = s * r[at_pole]
        dz[at_pole] = s * np.cos(deg2rad*(za[at_pole]))
        dx[at_pole] = np.sin(deg2rad*(za))
        dy[at_pole] = dx[at_pole] * np.sin(deg2rad*(aa))
        dx[at_pole] *= np.cos(deg2rad*(aa))

    not_pole = np.logical_not(at_pole)

    if any(not_pole):
        latrad = deg2rad * lat[not_pole]
        lonrad = deg2rad * lon[not_pole]
        zarad = deg2rad * za[not_pole]
        aarad = deg2rad * aa[not_pole]

        coslat = np.cos(latrad)
        sinlat = np.sin(latrad)
        coslon = np.cos(lonrad)
        sinlon = np.sin(lonrad)
        cosza = np.cos(zarad)
        sinza = np.sin(zarad)
        cosaa = np.cos(aarad)
        sinaa = np.sin(aarad)

        x[not_pole] = r[not_pole] * coslat
        y[not_pole] = x[not_pole] * sinlon
        x[not_pole] *= coslon
        z[not_pole] = r[not_pole] * sinlat

        dr = cosza
        dlat = sinza * cosaa
        dlon = sinza * sinaa / coslat

        dx[not_pole] = (coslat * coslon * dr - sinlat * coslon * dlat -
                        coslat * sinlon * dlon)
        dz[not_pole] = sinlat * dr + coslat * dlat
        dy[not_pole] = (coslat * sinlon * dr - sinlat * sinlon * dlat +
                        coslat * coslon * dlon)

    return x, y, z, dx, dy, dz


def get_ellipsoid_semiminor_axis(ellipsoid):
    """Returns the semiminor axis for the ellipsoid
    """
    return ellipsoid[0] * np.sqrt(1.0 - ellipsoid[1]**2)


def line_ellipsoid_intersect(x, y, z, dx, dy, dz,
                             ellipsoid, altitude=0.0):
    """ Finds positions of intersection of line with ellipsoid

    Solves for d:
        $$(X/a)**2 + (Y/a)**2 + (Z/b)**2 - 1 = 0$$
        $$X = x + d * dx$$
        $$Y = y + d * dy$$
        $$Z = z + d * dz$$
    where d is a multiplier of sqrt(dx**2 + dy**2 + dz**2), a is the
    semimajor axis and b is the semiminor axis. X, Y, Z gives the
    point of intersection.  If dx, dy, dz is normalized, d is distance
    in ellipsoid size units.  If no intersect, the code returns np.nan distance

    For real d, the halfway point between the points is the tangent point,
    though it is below the ellipsoid + altitude surface

    Returns:
        d: Array of distance parameters, last dimension gives the 2 solutions

    Parameters:
        x: Geocentric cartesian x-coordinate

        y: Geocentric cartesian y-coordinate

        z: Geocentric cartesian z-coordinate

        dx: Geocentric cartesian dx-view

        dy: Geocentric cartesian dy-view

        dz: Geocentric cartesian dz-view

        elliposid: ellipsoid model [a, e]

        altitude: altitude above elipsoid to intersect at [defaults to zero]

    Todo:
        Refer to line_sphere_intersect if ellipsoid is sphere
    """

    # semimajor axis
    a = ellipsoid[0] + altitude

    # semiminor axis
    b = get_ellipsoid_semiminor_axis(ellipsoid) + altitude

    # If these are scalars make them arrays
    x, y, z, dx, dy, dz = _broadcast(x, y, z, dx, dy, dz)

    # A*d**2 + B*d + C = 0, solve for d
    A = ((dx**2 + dy**2) / a**2 + dz**2 / b**2).flatten()
    B = (2 * ((x * dx + y * dy) / a**2 + z * dz / b**2)).flatten()
    C = ((y**2 + x**2) / a**2 + z**2 / b**2 - 1.0).flatten()

    d = np.zeros((len(A), 2))
    for i in range(len(A)):
        roots = np.roots([A[i], B[i], C[i]])
        if any(np.isreal(roots)):
            d[i, 0] = roots[0]
            d[i, 1] = roots[1]
        else:
            d[i, 0] = np.nan
            d[i, 1] = np.nan

    sh = []
    for i in x.shape:
        sh.append(i)
    sh.append(2)

    return d.reshape(tuple(sh))


def geometric_limb_zenith_angle(ellipsoid, r, lat, lon,
                                aa=0.0, alt=0.0, za_acc=1e-10):
    """Numerical method to compute the geometric limb zenith angle knowing
    other satellite parameters

    Assumes the planet is an ellipsoid and calculates the limb by numerically
    minimizing the distance between adjacent intersections of the ellipsoid.

    The rate of closing in on the right zenith angle should follow 180/30**N,
    though numerical errors at very exact accuracies are not guarded against.

    Parameters:

        ellipsoid: ellipsoid [a, e]

        r: geocentric radius of satellite

        lat: geocentric latitude of satellite

        lon: geocentric longitude of satellite

        aa: azimuth angle of satellite

        alt: tangent altitude

        za_acc: zenith angle accuracy.  Warning: Endless loop if too small

    Returns:

        za: satellite zenith angle for limb view geometry. Shaped as input
    """

    if(za_acc == 0):
        raise RuntimeError("Zenith accuracy cannot be 0")

    r, lat, lon, aa, alt = _broadcast(r, lat, lon, aa, alt)

    # Remember shape because we'll flatten these arrays for easy looping
    sh = r.shape
    r = r.flatten()
    lat = lat.flatten()
    lon = lon.flatten()
    aa = aa.flatten()
    alt = alt.flatten()

    za = np.empty_like(r)

    for i in range(len(r)):
        za_min = 0.0
        za_max = 180.0

        while True:  # Until zenith accuracy is reached

            # num=31 means 180/30**N is the number of loops to reach accuracy
            zenith_angles = np.linspace(za_min, za_max, num=31)

            # Get the Cartesian views for the probing zenith accuracies
            x, y, z, dx, dy, dz = geocentricposlos2cart(r[i], lat[i], lon[i],
                                                        zenith_angles, aa[i])

            # Get intersections with the ellipsoid at the tangent altitude
            d = line_ellipsoid_intersect(x, y, z, dx, dy, dz,
                                         ellipsoid, alt[i])

            # If we have somehow designed a scenario that is impossible or bad
            if np.isnan(d).all():
                raise RuntimeError("Not possible to get a tangent point.  " +
                                   "All tries miss the ellipsoid.")
            elif not np.isnan(d).any():
                raise RuntimeError("Not possible to get a tangent point.  " +
                                   "Inside ellipsoid or atmosphere.")

            # Limb zenith from minimizing distance between intersections
            D = d[:, 0] - d[:, 1]

            # These are the points of interest
            these = np.logical_and(D > 0, np.logical_not(np.isnan(D)))

            # The first of the zenith angles hitting the atmosphere is updated
            za_max = zenith_angles[these][0]

            # The last of the zenith angles missing the atmosphere is updated
            za_min = zenith_angles[np.logical_not(these)][-1]

            # Are we there yet?
            if (za_max-za_min) < za_acc:
                break

        # Note assignment to flat zenith angle array
        za[i] = za_max

    # Note that we return with the proper shape
    return za.reshape(sh)


def sphere_plane_intersection(pos, r, theta=np.linspace(-180., 180.)):
    """Computes the intersection of a plane with a sphere

    Returns cartesian x, y, z at the provided angles.  Sphere is centered
    at 0, 0, 0

    The solution is from

    .. math::
        \\vec{r}_c = \\vec{r}_p + r_0 \\left( \\vec{v}_1\\cos\\theta +
        \\vec{v}_2\\sin\\theta \\right)

    rp is the input position of the center of the interesecting circle,

    .. math::
        r_0 = \\sqrt{r^2 - \\vec{r_p} \\cdot \\vec{r_p}},\\;\\;
        \\vec{n} = \\vec{r}_p/||\\vec{r}_p||,\\;\\;
        \\vec{n}\\cdot\\vec{v}_1\\approx0,\\;\\;
        \\vec{v}_2 = \\vec{n} \\times \\vec{v}_1

    where v1 is so defined that it fully overlaps with the azimuth angle at the
    equator.  At other latitudes, the azimuth and theta are only the same at 0
    degrees.

    Parameters:
        pos (N, 3-array): position at the center of the circle-like shape that
        the intersection will form (i.e., the tangent point of a satellite at
        the sphere's radius).  Cannot be the 0-vector

        r (float): radius of sphere in same units as pos. Cannot be 0.0

        theta (n-array): determines positions by angle [degrees]

    Returns:
        (n, N, 3)-array of points intersecting plane and sphere

    Examples:
        Find possible positions that a satellite at an orbit about
        600 km above the surface can see a tangent point 30 km above
        the surface of Earth

        >>> x, y, z = geodetic2cart(30e3, 30., 30.)
        >>> sat = sphere_plane_intersection(np.array([x, y, z]), 6978e3)
    """
    assert r > 0.0, "Cannot work with zero radius sphere"

    assert len(pos.shape) == 2, "Need (N, 3) as pos-shape"

    d = np.linalg.norm(pos, axis=1)
    assert all(d > 0.0), "Cannot work with the 0-vector for position"

    number_of_angles = len(theta)
    number_of_positions = pos.shape[0]
    points = np.empty((number_of_angles, number_of_positions, 3))
    if number_of_angles == 0:
        return points

    # If the sphere is under the plane, or tangential to the plane
    for ii in range(number_of_positions):
        if d[ii] > r:
            for i in range(number_of_angles):
                points[i, ii, :] = np.array([np.nan, np.nan, np.nan])
            continue
        elif d[ii] == r:
            for i in range(number_of_angles):
                points[i, ii, :] = pos[ii, :]
            continue

        # The (normalized/unit) plane vector
        n = pos[ii] / d[ii]

        # Southern vector
        ones, lat, lon = cart2geocentric(n[0], n[1], n[2])

        if(lat == 90.0):
            v1 = np.array([cosd(lon), sind(lon), 0.]) * sind(lat)
        else:
            v1 = np.array([cosd(lon) * sind(lat),
                           sind(lon) * sind(lat),
                           cosd(lat-180.0)])

        # Eastern vector
        v2 = -np.cross(n, v1)  # points East

        # Radius of the circle
        rc = np.sqrt(r**2 - d[ii]**2)

        for i in range(number_of_angles):
            points[i, ii, :] = pos[ii, :] + rc * (cosd(theta[i]) * v1 +
                                                  sind(theta[i]) * v2)
    return points


def _broadcast(*args):
    """ Similar to broadcast_arrays in numpy but with minimum output size (1,)
    """
    shape = np.broadcast(*args).shape
    if not shape:
        shape = (1,)
    return [np.broadcast_to(array, shape) for array in args]
