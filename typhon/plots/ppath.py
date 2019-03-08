# -*- coding: utf-8 -*-

"""Functions to create plots using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    'plot_ppath',
    'plot_ppath_field',
    'ppath_field_minmax_posbox',
    'adjust_ppath_field_box_by_minmax',
    'plot_ppath_field_zenith_coverage_per_gp_p',
]


def plot_ppath(ax, ppath, planetary_radius=1, lat_is_x=True, scale_alt=1000):
    """ Return the ARTS Ppath plotted on the surface
    
    Parameters:
        ax:
            (AxesSubplot): matplotlib axes to plot into.  Assumes polar project
        ppath:
            (Ppath or Workspace)  Propagation path or Wosrkspace with the path
        planetary_radius:
            (Numeric)  Spherical radius of the planetary body (if ppath_field is not Workspace)
        lat_is_x:
            (Boolean) Have lat or lon act as the x-coordinate?
        scale_alt:
            (Numeric) Divide altitude and planetary_radius by this value
    """
    
    try:
        ppath = ppath.ppath.value
    except:
        pass
    
    try:
        planetary_radius = ppath.refellipsoid.value[0]
    except:
        pass
    
    alt, lat, lon, za, aa = ppath.alt_lat_lon_za_aa()
    
    if lat_is_x:
        x = lat
    else:
        x = lon
    
    ax.plot(np.deg2rad(x), alt/scale_alt)
    ax.set_rorigin(-planetary_radius/scale_alt)


def plot_ppath_field(ppath_field, planetary_radius=1, lat_is_x=True,
                     scale_alt=1000, subplots=1):
    """ Return the ARTS Ppath plotted on the surface
    
    Parameters:
        ppath_field:
            (Ppath or Workspace)  Propagation path or Wosrkspace with the path
        planetary_radius:
            (Numeric)  Spherical radius of the planetary body (if ppath_field is not Workspace)
        lat_is_x:
            (Boolean) Have lat or lon act as the x-coordinate?
        scale_alt:
            (Numeric) Divide altitude and planetary_radius by this value
        subplots:
            (Index)  Divides the ppath_field into this many equally long
            segments, so that the first len(ppath_field) // subplots plots
            are drawn on the same surface and so forth
    
    Returns:
        list of axis of the length of subplots
    """
    
    try:
        ppath_field = ppath_field.ppath_field.value
    except:
        pass
    
    try:
        planetary_radius = ppath_field.refellipsoid.value[0]
    except:
        pass
    
    N = len(ppath_field)
    if N % subplots != 0:
        raise RuntimeError("Bad inputs. Only for evenly divisible numbers")
    
    # Number of drawings per subplot
    n = N // subplots
    path_ind = 0
    
    n1 = int(np.ceil(np.sqrt(subplots)))
    n2 = n1 if np.isclose(n1*n1, subplots) or n1*n1 > subplots else n1 + 1
    
    axes = []
    for i in range(subplots):
        axes.append(plt.subplot(n1, n2, i+1, projection='polar'))
        for j in range(n):
            plot_ppath(axes[-1], ppath_field[path_ind], 
                       planetary_radius, lat_is_x, scale_alt)
            path_ind += 1
    return axes


def ppath_field_minmax_posbox(ppath_field):
    """ Return the minimum and maximum of all pos variables of ppath_field
    
    Parameters:
        ppath_field:
            (Ppath or Workspace)  Propagation path or Wosrkspace with the path
    
    Returns:
        min(alt), max(alt), min(lat), max(lat), min(lon), max(lon)
    """
    
    try:
        ppath_field = ppath_field.ppath_field.value
    except:
        pass
    
    minalt = minlat = minlon = np.infty
    maxalt = maxlat = maxlon = -np.infty
    
    for ppath in ppath_field:
        alt, lat, lon, za, aa = ppath.alt_lat_lon_za_aa()
        
        minalt = min(alt.min(), minalt)
        maxalt = max(alt.max(), maxalt)
        
        minlat = min(lat.min(), minlat)
        maxlat = max(lat.max(), maxlat)
        
        minlon = min(lon.min(), minlon)
        maxlon = max(lon.max(), maxlon)
        
    return minalt, maxalt, minlat, maxlat, minlon, maxlon


def adjust_ppath_field_box_by_minmax(axes, ppath_field, lat_is_x=True,
                                     scale_alt=1000, extrap=0.05):
    """ Adjusts the axis of plotted ppath_field to 
    
    Parameters:
        axes:
            Plotted axes of ppath_field
        ppath_field:
            (Ppath or Workspace)  Propagation path or Wosrkspace with the path
        planetary_radius:
            (Numeric)  Spherical radius of the planetary body (if ppath_field is not Workspace)
        lat_is_x:
            (Boolean) Have lat or lon act as the x-coordinate?
        scale_alt:
            (Numeric) Divide altitude and planetary_radius by this value
        extrap:
            (Numeric) How large an extrapolation of the box in terms of full
            posbox minmax
    """
    
    minalt, maxalt, minlat, maxlat, minlon, maxlon = ppath_field_minmax_posbox(ppath_field)
    
    if lat_is_x:
        xmin = minlat
        xmax = maxlat
        max_diff = 180
    else:
        xmin = minlon
        xmax = maxlon
        max_diff = 360
    dx = (xmax-xmin) * extrap
    
    xmin -= dx
    xmax += dx
    
    if (xmax-xmin) > max_diff:
        return  # no adjustment possible
    
    for ax in axes:
        ax.set_thetamin(xmin)
        ax.set_thetamax(xmax)


def wzeniths(zeniths):
    n = len(zeniths)
    if not n:
        return np.array([0, 180]), np.array([1, 1])
    inds = np.argsort(zeniths)
    zaz = np.deg2rad(zeniths[inds])
    cz = np.cos(zaz)
    
    wz = np.zeros((3*n))
    za = np.zeros((3*n))
    for i in range(n-1):
        N = i*3
        za[0+N] = zaz[i]
        za[1+N] = 0.5 * (zaz[i] + zaz[i+1])
        za[2+N] = zaz[i+1]
        
        w = 0.5 * (cz[i] - cz[i+1])
        wz[0+N] = wz[1+N] = wz[2+N] = w
    
    za[-1] = np.deg2rad(180)
    return za, wz


def plot_ppath_field_zenith_coverage_per_gp_p(ppath_field, scale_alt=1000):
    """Plots the zenith angle coverage of a ppath_field for all the altitudes
    in the field.
    
    Parameters:
        ppath_field:
            (Ppath or Workspace)  Propagation path or Wosrkspace with the path
        scale_alt:
            (Numeric) Divide altitude and planetary_radius by this value
    
    Returns:
        list of axis, list of altitudes, list of maximum weights
    """
    try:
        ppath_field = ppath_field.ppath_field.value
    except:
        pass
    
    alt = []
    gps = []
    for path in ppath_field:
        for i in range(path.np):
            if path.gp_p[i] not in gps:
                gps.append(path.gp_p[i])
                alt.append(path.pos[i, 0])
    alt = np.array(alt)
    
    # Size of subplots
    N = len(alt)
    
    zas = np.full((N), None)
    for path in ppath_field:
        for i in range(path.np):
            ip = np.where(path.pos[i, 0] == alt)[0][0]
            if zas[ip]:
                zas[ip].append(path.los[i, 0])
            else:
                zas[ip] = [path.los[i, 0]]
            
    # Sub-plot grid
    n1 = int(np.ceil(np.sqrt(N)))
    n2 = n1 if np.isclose(n1*n1, N) or n1*n1 > N else n1 + 1
    
    inds = np.argsort(alt)
    
    axes = []
    wei = []
    alts = []
    for i in range(N):
        pos = inds[i]
        za, wz = wzeniths(np.array(zas[pos]))
        
        wei.append(wz.max())
        alts.append(alt[pos])
        
        axes.append(plt.subplot(n1, n2, i+1, projection='polar'))
        axes[-1].plot(za, wz,'k')
        axes[-1].set_rorigin(-0.25/np.pi)
        axes[-1].set_thetamin(0)
        axes[-1].set_thetamax(180)
        axes[-1].set_rgrids([0, wz.max()*1.05], ['', ''])
        axes[-1].set_thetagrids([0, 90, 180], ['Zenith', 'Limb', 'Nadir'])
        axes[-1].set_title("Alt {} km".format(alt[pos]/scale_alt))
    
    return axes, alts, wei
