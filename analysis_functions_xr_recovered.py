# analysis functions that work for xarray dataarrays and datasets
import numpy as np 
import xarray as xr 

def calc_tidal_av(coordinates : xr.DataArray , window :int) -> xr.DataArray:
    """
    calculate tidal average signal over window steps. Depends on:
    - coordinates:xr dataarray with dimensions trajectory and obs
    - window: (int) number of succesive observations over the coordinates
      wil be averagd.

    """
    cs = coordinates.cumsum(dim='obs',skipna=False)
    mean = (cs-cs.roll(obs=window))/float(window)
    return mean

def Haversine_list(lon : xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """ this function calculates the path length in km of a between 2 points using Haversine formula (https://en.wikipedia.org/wiki/Haversine_formula)
    """      

    mean_radius_earth = 6371 #mean readius earth in km
    deg2rad=np.pi/180
    londif = lon.diff(dim='obs')#np.diff(lon)
    latdif = lat.diff(dim='obs')
    d = 2 * mean_radius_earth * np.arcsin( np.sqrt(np.sin(0.5*latdif*deg2rad)**2 + np.cos(lat[:-1]*deg2rad) * np.cos(lat[1:]*deg2rad)* np.sin(0.5*londif*deg2rad)**2 ))
    return d

def trajectory_length(lon : xr.DataArray,lat : xr.DataArray) -> xr.DataArray:
    """
    This function calculates the along track length/trajectory length in km
    """
    
    d = Haversine_list(lon, lat)
    traj=d.cumsum(dim='obs',skipna=False)
    return traj


def derivative_backward(var : xr.DataArray, dt: float) -> xr.DataArray:
    varmin = var.shift(obs=-1)
    return (var-varmin)/(1 * dt)

def derivative_forward(var : xr.DataArray, dt: float) -> xr.DataArray:
    varplus = var.shift(obs=1)
    return (varplus-var)/(1 * dt)

def derivative_middle(var : xr.DataArray, dt: float) -> xr.DataArray:
    varmin = var.shift(obs=-1)
    varplus = var.shift(obs=1)
    return (varplus-varmin)/(2 * dt)

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



def skill_score(lon1, lat1, lon2, lat2):
    dsn = Haversine(lon1,lat1,lon2,lat2).cumsum(dim='obs',skipna=False)
    dsn[:,0] =0
    dist1_diff = Haversine(lon1.roll(obs=1),lat1.roll(obs=1),lon1,lat1)
    # dist1_diff[:,0]=
    dln = dist1_diff.cumsum(dim='obs').cumsum(dim='obs',skipna=False)
    dln = np.where(dln < 1, np.nan,dln)
    return  1-dsn/dln


def normalized_separation_dist(data1, data2):
    
    lon1 = data1.lon 
    lat1 = data1.lat
    lon2 = data2.lon 
    lat2 = data2.lat
    lon_start = lon2.isel(obs=0)
    lat_start = lon2.isel(obs=0)
    relative_dist = Haversine(lon1,lat1,lon2,lat2)
    total_dist = Haversine(lon2,lat2,lon_start,lat_start)
    total_dist[:,0]=1
    return  relative_dist/total_dist