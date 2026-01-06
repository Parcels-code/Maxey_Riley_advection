"""
Standard analysis functions
"""

# import packages
import numpy as np
import xarray as xr
import warnings


def autocorr_time_trajectory(array):
    NPART, T = array.shape
    mean = np.mean(array)
    darray = array - mean
    acf = np.zeros((NPART, int(T / 2)))
    # nbins=np.zeros(T)

    for t in range(int(T / 2)):
        for dt in range(0, int(T / 2)):
            acf[:, dt] += darray[:, t] * darray[:, t + dt]

    # acf/=nbins

    acfmean = np.mean(acf, 0)
    return acfmean / acfmean[0]


def Haversine(lon1, lat1, lon2, lat2):
    """
    Function to calculate the path lenth in km between 2 points using the
    Haversine formula. It takes as input:
    - lon1: longitude of point 1 in degrees
    - lat1: latitude of point 1 in degrees
    - lon2: longitude of point 2 in degrees
    - lon2: latitude of point 2 in degrees

    Source: https://en.wikipedia.org/wiki/Haversine_formula
    """
    mean_radius_earth = 6371
    deg2rad = np.pi / 180
    arg = (
        np.sin(0.5 * (lat2 - lat1) * deg2rad) ** 2
        + np.cos(lat2 * deg2rad)
        * np.cos(lat1 * deg2rad)
        * np.sin(0.5 * (lon2 - lon1) * deg2rad) ** 2
    )
    d = 2 * mean_radius_earth * np.arcsin(np.sqrt(arg))
    return d


def zonal_dist(lon1, lon2, lat):
    """
    Function that calculates the zonal distance between point1 and point 2
    with sign in km, where the + is from west to east and - is from east
    to west. It takes the following input:
    - lon1: the longitude (in degrees) of the point 1
    - lon2: the longitude (in degrees) of the point 2
    - lat: the reference latitude (in degrees) of the 2 points
    """
    d = Haversine(lon1, lat, lon2, lat)
    dif = lon2 - lon1
    return d * np.sign(dif)


def meridional_dist(lat1, lat2):
    """
    Function that calculates the meridonal distnce with a sign in km
    where the + is from south to north and - is from north to south
    It takes the following input:
    - lat1: the latitude (in degrees) of point 1
    - lat2: the latitude (in degrees) of point 2
    """
    R = 6371.0  # mean radius of earth in km
    deg2rad = np.pi / 180
    dif = lat2 - lat1
    return R * dif * deg2rad


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def Haversine_list(lon, lat):
    """this function calculates the path length in km of a between 2 points using Haversine formula (https://en.wikipedia.org/wiki/Haversine_formula)"""
    if type(lon) == xr.DataArray:
        lon = lon.values
    if type(lat) == xr.DataArray:
        lat = lat.values
    mean_radius_earth = 6371  # mean readius earth in km
    deg2rad = np.pi / 180
    if lon.ndim == 1:
        londif = np.diff(lon)
        latdif = np.diff(lat)
        d = (
            2
            * mean_radius_earth
            * np.arcsin(
                np.sqrt(
                    np.sin(0.5 * latdif * deg2rad) ** 2
                    + np.cos(lat[:-1] * deg2rad)
                    * np.cos(lat[1:] * deg2rad)
                    * np.sin(0.5 * londif * deg2rad) ** 2
                )
            )
        )
    else:
        londif = np.diff(lon, axis=1)
        latdif = np.diff(lat, axis=1)
        d = (
            2
            * mean_radius_earth
            * np.arcsin(
                np.sqrt(
                    np.sin(0.5 * latdif * deg2rad) ** 2
                    + np.cos(lat[:, :-1] * deg2rad)
                    * np.cos(lat[:, 1:] * deg2rad)
                    * np.sin(0.5 * londif * deg2rad) ** 2
                )
            )
        )

    return d


def trajectory_length(lon, lat):
    """
    This function calculates the along track length/trajectory length in km
    """

    d = Haversine_list(lon, lat)
    if lon.ndim == 1:
        traj = np.cumsum(d)
    else:
        traj = np.cumsum(d, axis=1)
    return traj


# def make_PDF(x,nbins,norm,min=100000,max=-100000 ):
#     # remove nans
#     x =x[~np.isnan(x)]
#     if(min==100000):
#         min=np.nanmin(x)
#     if(max==-100000):
#         max=np.nanmax(x)

#     dx=(max-min)/nbins
#     max+=dx
#     min-=dx
#     indices=((x-min)/dx).astype(int)
#     pdf=np.zeros(nbins+3)
#     np.add.at(pdf,indices,1)
#     bins=np.arange(min,max+0.1*dx,dx)+0.5*dx
#     if(norm == True):
#         pdf/=x.size*dx
#     return bins, pdf


def make_PDF(x: np.array, nbins: int, norm: bool, vmin=None, vmax=None):
    # check typess

    # Set min/max only if not provided
    vmin = np.nanmin(x) if vmin is None else vmin
    vmax = np.nanmax(x) if vmax is None else vmax

    if (x > vmax).any():
        warnings.warn(
            "Some values in x are greater than vmax, thus not all values are used in the pdf",
            UserWarning,
        )
    if (x < vmin).any():
        warnings.warn(
            "Some values in x are smaller than vmin, thus not all values are used in the pdf",
            UserWarning,
        )
    dx = (vmax - vmin) / nbins
    pdf, bin_edges = np.histogram(x, bins=nbins, range=(vmin, vmax))
    bins = bin_edges[:-1] + 0.5 * dx
    bin_widths =bin_edges[1:]-bin_edges[:-1]
    if norm == True:
        pdf = pdf / x.size
    pdf = pdf/bin_widths 
    return bins, pdf


def make_lognormal_PDF(x: np.array, nbins: int, norm: bool, vmin=None, vmax=None):
    # Set min/max only if not provided
    vmin = np.nanmin(x) if vmin is None else vmin
    vmax = np.nanmax(x) if vmax is None else vmax

    # Ensure all values are positive (log-space requires x > 0)
    if (vmin <= 0):
        raise ValueError("x array has value <= O not possible for log distribution")
    x = x[x > 0]

    
  

    if (x > vmax).any():
        warnings.warn(
            "Some values in x are greater than vmax, thus not all values are used in the pdf",
            UserWarning,
        )
    if (x < vmin).any():
        warnings.warn(
            "Some values in x are smaller than vmin, thus not all values are used in the pdf",
            UserWarning,
        )

    # Generate logarithmically spaced bins
    bin_edges = np.logspace(np.log10(vmin), np.log10(vmax), nbins + 1)

    # Compute histogram
    pdf, _ = np.histogram(x, bins=bin_edges)

    # Compute bin centers in log-space
    bins = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    bin_widths =bin_edges[1:]-bin_edges[:-1]
    if norm:
        pdf = pdf / x.size
    pdf = pdf/bin_widths 
    return bins, pdf


def mean_angle(angles, units="deg"):
    """
    when calculating the mean angle one has to take into account angles wrap around
    for this the following formula is used:
     <theta> = argctan2(1/n Sum_j sin(theta_j), 1/n Sum_j cos(theta_j) )
     source:
     https://rosettacode.org/wiki/Averages/Mean_angle
    """
    if units == "deg":
        deg2rad = np.pi / 180
    elif units == "rad":
        deg2rad = 1
    else:
        raise TypeError("Unit of angle has to be deg (degrees) or rad (radians)")

    N = angles.size
    meanangle = np.arctan2(
        1 / N * np.sum(np.sin(angles * deg2rad)),
        1 / N * np.sum(np.cos(angles * deg2rad)),
    )
    return meanangle / deg2rad


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
