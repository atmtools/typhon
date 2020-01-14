# -*- coding: utf-8 -*-
"""Statistical functions for binary cloud masks. """
import numpy as np
import scipy as sc

from skimage import measure
from scipy.spatial.distance import pdist


__all__ = [
    'get_cloudproperties',
    'neighbor_distance',
    'iorg',
    'scai',
    'cloudfraction',
]


def get_cloudproperties(cloudmask, connectivity=1):
    """Calculate basic cloud properties from binary cloudmask.

    Note:
        All parameters are calculated in pixels!!

    See also:
        :func:`skimage.measure.label`:
            Used to find different clouds. 
        :func:`skimage.measure.regionprops`:
            Used to calculate cloud properties.

    Parameters:
        cloudmask (ndarray): 2d binary cloud mask.
        connectivity (int):  Maximum number of orthogonal hops to consider
            a pixel/voxel as a neighbor (see :func:`skimage.measure.label`).

    Returns:
        list:
            List of :class:`RegionProperties`
            (see :func:`skimage.measure.regionprops`)
    """
    cloudmask[np.isnan(cloudmask)] = 0

    labels = measure.label(cloudmask, connectivity=connectivity)

    return measure.regionprops(labels)


def neighbor_distance(cloudmask, connectivity=1):
    """Calculate nearest neighbor distance for each cloud.

    Note: 
        Distance is given in pixels.

    See also: 
        :class:`scipy.spatial.cKDTree`:
            Used to calculate nearest neighbor distances. 

    Parameters: 
        cloudmask (ndarray): 2d binary cloud mask.
        connectivity (int):  Maximum number of orthogonal hops to consider
            a pixel/voxel as a neighbor (see :func:`skimage.measure.label`).

    Returns: 
        ndarray: Nearest neighbor distances in pixels.
    """
    cloudproperties = get_cloudproperties(cloudmask, connectivity=connectivity)
    
    centroids = [prop.centroid for prop in cloudproperties]
    indices = np.arange(len(centroids))
    neighbor_distance = np.zeros(len(centroids))
    centroids_array = np.asarray(centroids)

    for n, point in enumerate(centroids):
        # use all center of mass coordinates, but the one from the point
        mytree = sc.spatial.cKDTree(centroids_array[indices != n])
        dist, indexes = mytree.query(point)
        neighbor_distance[n] = dist

    return neighbor_distance


def iorg(cloudmask, connectivity=1):
    """Calculate the cloud cluster index 'I_org'.

    See also: 
        :func:`scipy.integrate.trapz`:
            Used to calculate the integral along the given axis using
            the composite trapezoidal rule.

    Parameters: 
        cloudmask (ndarray): 2d binary cloud mask.
        connectivity (int):  Maximum number of orthogonal hops to consider
            a pixel/voxel as a neighbor (see :func:`skimage.measure.label`).

    Returns:
        float: cloud cluster index I_org.

    References: 
        Tompkins, A. M., and A. G. Semie (2017), Organization of tropical 
        convection in low vertical wind shears: Role of updraft entrainment, 
        J. Adv. Model. Earth Syst., 9, 1046–1068, doi: 10.1002/2016MS000802.
        
    """
    nn = neighbor_distance(cloudmask, connectivity=connectivity)
    nn_sorted = np.sort(nn)
    
    nncdf = np.linspace(0, 1, len(nn))
    
    # theoretical nearest neighbor cumulative frequency
    # distribution (nncdf) of a random point process (Poisson)
    lamb = nn.size / cloudsmask.size
    nncdf_poisson = 1 - np.exp(-lamb * np.pi * nn_sorted**2)

    return sc.integrate.trapz(y=nncdf, x=nncdf_poisson)


def scai(cloudmask, connectivity=1):
    """Calculate the 'Simple Convective Aggregation Index (SCAI)'.  

    The SCAI is defined as the ratio of convective disaggregation
    to a potential maximal disaggregation.

    See also: 
        :func:`scipy.spatial.distance.pdist`:
            Used to calculate pairwise distances between cloud entities. 
        :func:`scipy.stats.mstats.gmean`:
            Used to calculate the geometric mean of all clouds in pairs. 

    Parameters:
        cloudmask (ndarray): 2d binary cloud mask.
        connectivity (int):  Maximum number of orthogonal hops to consider
            a pixel/voxel as a neighbor (see :func:`skimage.measure.label`).

    Returns:
        float: SCAI.

    References: 
        Tobin, I., S. Bony, and R. Roca, 2012: Observational Evidence for 
        Relationships between the Degree of Aggregation of Deep Convection, 
        Water Vapor, Surface Fluxes, and Radiation. J. Climate, 25, 6885–6904,
        https://doi.org/10.1175/JCLI-D-11-00258.1

    """
    cloudproperties = get_cloudproperties(cloudmask, connectivity=connectivity)
    centroids = [prop.centroid for prop in cloudproperties]

    # number of cloud clusters
    N = len(centroids)

    # potential maximum of N depending on cloud connectivity
    N_max = np.sum(~np.isnan(cloudmask)) / 2
    if connectivity == 2:
        N_max = np.sum(~np.isnan(cloudmask)) / 4

    # distance between points (center of mass of clouds) in pairs
    di = pdist(centroids, 'euclidean')
    # order-zero diameter
    D0 = sc.stats.mstats.gmean(di)

    # characteristic length of the domain (in pixels): diagonal of box
    L = np.sqrt(cloudmask.shape[0]**2 + cloudmask.shape[1]**2)

    return N / N_max * D0 / L * 1000


def cloudfraction(cloudmask):
    """Calculate cloud fraction based on cloud mask, while irnoring NaNs.
    
    Parameters:
        cloudmask (ndarray): 2d binary cloud mask.
        
    Returns:
        float: cloud fraction.
    """
    return np.nansum(cloudmask) / np.sum(~np.isnan(cloudmask))
