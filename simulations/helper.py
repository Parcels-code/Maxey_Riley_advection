import os 
import numpy as np
import math
from datetime import datetime, timedelta
import xarray as xr

def create_filelist(input_directory : str, str : str ,time_start : datetime ,time_end : datetime,dt : timedelta, dt_name: timedelta) -> list:
    """
    Function that creates list of all input files between time_start and time_end for NWSHELF_ANALYSISFORECAST_PHY_004_013 hourly dataset from
    copernicus marine service (https://doi.org/10.48670/moi-00054). 
    - input_directory is a string of the directory where the files are stored
    - str is the basename of the files (with _t is current time and _tp is plus the interval given by dt_name)
    - dt is timestepping of files
    - dt_name is timestepping used in name of file
    """
    time = time_start
    files = []
    while (time<=time_end):
        time_tplus = time+dt_name
        y_t =time.year
        m_t = time.month
        d_t = time.day
        y_tp = time_tplus.year
        m_tp=time_tplus.month
        d_tp=time_tplus.day
        files.append(input_directory+str.format(year_t = y_t, month_t = m_t, day_t = d_t, year_tplus = y_tp, month_tplus = m_tp, day_tplus =d_tp))
        time += dt
        
    return files

def displace_coordinates(lon : np.array , lat: np.array , d : float, B: float) -> tuple[np.array, np.array]:
    """
    Function that displaces point(s) given by lon, lat over a distance d
    (in meters) in direction B (angle measured clockwise in radians from the
    north pole). The function returns the lon and lat coordinates of the
    displaced point(s).
    """
    Rearth = 6371 * 10**3 # radius earth in m
    lon_rad = lon * np.pi/180. 
    lat_rad = lat * np.pi/180. 
    lat_new = np.arcsin(np.sin(lat_rad) * np.cos(d / Rearth)+np.cos(lat_rad) * np.sin( d/ Rearth) * np.cos(B))
    lon_new = lon_rad + np.arcsin( np.sin(d / Rearth) * np.sin(B) / np.cos(lat_new))
    lat_new_angle = lat_new * 180/np.pi 
    lon_new_angle = lon_new * 180/np.pi 
    return lon_new_angle, lat_new_angle

def getclosest_ij(lats : np.array , lons : np.array , latpt: np.array, lonpt: np.array) -> np.array:
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats - latpt) ** 2 + (lons - lonpt) 
    minindex_flattened = np.nanargmin(dist_sq)  
    return np.unravel_index(minindex_flattened,
                            lats.shape)  


def set_particles_region(land_mask_file : str,lonmin: float,lonmax: float,latmin: float,latmax: float,name_lon : str = 'lon' , name_lat : str = 'lat')-> tuple[np.array,np.array]:
    """
    Function that creates lon and lat list of 1 particle per gridcell release taking into account 
    the region selection as given by longitudes between lonmin and lonmax and latitudes between
    latmin and latmax and removing all particles that are placed on land (using a seperately  created 
    landmask file)
    """
    data_mask=xr.open_dataset(land_mask_file)
    lon=data_mask[name_lon].data
    lat=data_mask[name_lat].data


    iy_min, ix_min = getclosest_ij(lat, lon, latmin, lonmin)
    iy_max, ix_max = getclosest_ij(lat, lon, latmax, lonmax)
    
    lon_region=lon[iy_min:iy_max,ix_min:ix_max]
    lat_region=lat[iy_min:iy_max,ix_min:ix_max]
    mask_region=data_mask['mask_land'][iy_min:iy_max,ix_min:ix_max]
    lon_selection=lon_region[np.where(mask_region==False)]
    lat_selection=lat_region[np.where(mask_region==False)]
    return lon_selection,lat_selection
