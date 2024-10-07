import os 
import numpy as np
import math
from datetime import datetime, timedelta
import xarray as xr

def create_filelist(input_directory, str ,time_start,time_end,dt, dt_name):
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

def displace_coordinates(lon, lat, d, B):
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

def getclosest_ij(lats, lons, latpt, lonpt):
    """Function to find the index of the closest point to a certain lon/lat value."""
    dist_sq = (lats - latpt) ** 2 + (lons - lonpt) ** 2  # find squared distance of every point on grid
    minindex_flattened = np.nanargmin(dist_sq)  # 1D index of minimum dist_sq element
    return np.unravel_index(minindex_flattened,
                            lats.shape)  # Get 2D index for latvals and lonvals arrays from 1D index



def set_particles_region(land_mask_file,lonmin,lonmax,latmin,latmax):
    """
    Function that creates lon and lat list of 1 particle per gridcell release taking into account 
    the region selection as given by longitudes between lonmin and lonmax and latitudes between
    latmin and latmax and removing all particles that are placed on land (using a seperately  created 
    landmask file)
    TO DO: make it possible to release multiple plarticles per cell (think/ask about distribution?)
    """
    data_mask=xr.open_dataset(land_mask_file)
    lon=data_mask['lon'].data
    lat=data_mask['lat'].data


    iy_min, ix_min = getclosest_ij(lat, lon, latmin, lonmin)
    iy_max, ix_max = getclosest_ij(lat, lon, latmax, lonmax)
    
    lon_region=lon[iy_min:iy_max,ix_min:ix_max]
    lat_region=lat[iy_min:iy_max,ix_min:ix_max]
    mask_region=data_mask['mask_land'][iy_min:iy_max,ix_min:ix_max]
    lon_selection=lon_region[np.where(mask_region==False)]
    lat_selection=lat_region[np.where(mask_region==False)]
    return lon_selection,lat_selection
