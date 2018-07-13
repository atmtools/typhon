# -*- coding: utf-8 -*-
"""Functions for calculating cloud field statistics from binary cloud mask 
arrays.
"""
import numpy as np
import scipy as sc

from skimage import measure
from scipy.spatial.distance import pdist


__all__ = [
    'get_cloud_properties',
    'neighbor_distance',
    'iorg',
    'scai',
]


def get_cloud_properties(cloudmask, connectivity=4):
    """Calculate basic cloud properties from binary cloudmask.

    Note:
        All parameters are calculated in pixels!!

    See also:
        `skimage.measure.labels`: Used to find different clouds. 
        `skimage.measure.regionprops`: Used to calculate cloud properties.

    Parameters:
        cloudmask (ndarray): 2d binary cloud mask.
        connectivity (int):  Maximum number of orthogonal hops to consider
            a pixel/voxel as a neighbor (see `skimage.measure.labels`).

    Returns:
        list: List of `RegionProperties`
    """
    cloudmask[np.isnan(cloudmask)] = 0

    labels = measure.label(cloudmask, connecivity=connectivity)

    return measure.regionprops(labels)


def neighbor_distance(cloudproperties):
    """Calculate nearest neighbor distance for each cloud.

    Note: 
        distance is given in pixels!!

    See also: 
        `scipy.spatial.cKDTree`: Used to calculate nearest neighbor distances. 

    Parameters: 
        cloudproperties (object): Output of function get_cloud_properties. 

    Returns: 
        list: List of nearest neighbor distances
    """
    neighbor_distance = []
    centroids = [prop.centroid for prop in cloudproperties]

    for point in centroids:
        # use all center of mass coordinates, but the one from the point
        mytree = sc.spatial.cKDTree(np.asarray(
            centroids)[np.arange(len(centroids)) != centroids.index(point)])
        dist, indexes = mytree.query(point)
        neighbor_distance.append(dist)

    return neighbor_distance


def iorg(neighbor_distance, cloudmask):
    """Calculate the cloud cluster index 'I_org' following Tompkins and Semie, 
        2017. 

    See also: 
        `scipy.integrate.trapez`: Used to calculate the integral along the 
            given axis using the composite trapezoidal rule.

    Parameters: 
        neighbor_distance (list): nearest neighbor distances. 
            Output of function neighbor_distance. 
        cloudmask (ndarray): 2d binary cloud mask.

    Returns:
        float: cloud cluster index I_org.

    References: 
        Tompkins, A. M., and A. G. Semie (2017), Organization of tropical 
        convection in low vertical wind shears: Role of updraft entrainment, 
        J. Adv. Model. Earth Syst., 9, 1046–1068, doi: 10.1002/2016MS000802.
        
    """
    bins = np.arange(0, 200, 4)
    bins_mid = bins[:-1] + .5

    try:
        # nearest neighbor cumulative frequency distribution (nncdf)
        nncdf, bins = np.histogram(neighbor_distance, bins=bins, normed=True,
                                   cumulative=True)

        # theoretical nearest neighbor cumulative frequency
        # distribution (nncdf) of a random point process (Poisson)
        lamb = (len(neighbor_distance) /
                (cloudmask.shape[0] * cloudmask.shape[1]))
        nncdf_poisson = 1 - np.exp(-lamb * np.pi * bins_mid**2)

        iorg = sc.integrate.trapz(y=nncdf, x=nncdf_poisson)
    except:
        iorg = np.nan

    return iorg


def scai(cloudproperties, cloudmask, connectivity=4, NaNmask=None):
    """Calculate the cloud cluster index 'Simple Convective Aggregation 
        Index (SCAI)' following Tobin et al., 2012.  

    See also: 
        `scipy.spatial.distance.pdist`: Used to calculate pairwise distances 
            between cloud entities. 
        `scipy.stats.mstats.gmean`: Used to calculate the geometric mean of 
            all clouds in pairs. 

    Parameters:
        cloudproperties (object): Output of function get_cloud_properties. 
        cloudmask (ndarray): 2d binary cloud mask.
        connectivity (int):  Maximum number of orthogonal hops to consider
            a pixel/voxel as a neighbor (see `skimage.measure.labels`).
        NaNmask (ndarray): 2d mask of non valid edge pixels.

    Returns:
        float: ratio of convective disaggregation to a potential maximal 
            disaggregation.

    References: 
        Tobin, I., S. Bony, and R. Roca, 2012: Observational Evidence for 
        Relationships between the Degree of Aggregation of Deep Convection, 
        Water Vapor, Surface Fluxes, and Radiation. J. Climate, 25, 6885–6904,
        https://doi.org/10.1175/JCLI-D-11-00258.1

    """
    if NaNmask is None:
        NaNmask = np.ones(cloudmask.shape)

    centroids = [prop.centroid for prop in cloudproperties]

    # number of cloud clusters
    N = len(centroids)

    # potential maximum of N depending on cloud connectivity
    if connectivity == 4:
        chessboard = np.ones(cloudmask.shape).flatten()
        # assign every second element with "0"
        chessboard[np.arange(1, len(chessboard), 2)] = 0
        # reshape to original cloudmask.shape
        chessboard = np.reshape(chessboard, cloudmask.shape)
        # inlcude NaNmask
        chessboard[NaNmask == np.nan] = np.nan
        N_max = np.nansum(chessboard)
    elif connectivity == 8:
        chessboard[np.arange(1, cloudmask.shape[0], 2), :] = 0
        chessboard = np.reshape(chessboard, cloudmask.shape)
        chessboard[NaNmask == np.nan] = np.nan
        N_max = np.sum(chessboard)
    else:
        raise ValueError('Connectivity argument can only be {4,8}')

    # distance between points (center of mass of clouds) in pairs
    di = pdist(centroids, 'euclidean')
    # order-zero diameter
    D0 = sc.stats.mstats.gmean(di)

    # characteristic length of the domain (in pixels): diagonal of box
    L = np.sqrt(cloudmask.shpae[0]**2 + cloudmask.shape[1]**2)

    return N / N_max * D0 / L * 1000
