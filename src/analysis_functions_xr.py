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