"""
Script to calculate spatial and temporal derivative fields of the velocity
fields in the NWES. The derivative fields are saved to a netcdf file, which
can be used as input for simulations_NWES_derivative_fields.py 
"""


import xarray as xr
import numpy as np
import xgcm 
from datetime import datetime, timedelta
import sys
sys.path.append("/nethome/4291387/Maxey_Riley_advection/" \
"Maxey_Riley_advection/simulations")
from helper import create_filelist

#

starttimes = [datetime(2023,9,1,00,00,00),
             datetime(2023,10,1,00,00,00),
             datetime(2023,11,1,00,00,00),
             datetime(2023,12,1,00,00,00),
             datetime(2024,1,1,00,00,00),
             datetime(2024,2,1,00,00,00)]

# set directories and files
directory_in = ('/storage/shared/oceanparcels/input_data/'
                'CopernicusMarineService/'
                'NORTHWESTSHELF_ANALYSIS_FORECAST_PHY_004_013/')
input_filename_base = ('CMEMS_v6r1_NWS_PHY_NRT_NL_01hav3D_'
                        '{year_t:04d}{month_t:02d}{day_t:02d}_'
                        '{year_t:04d}{month_t:02d}{day_t:02d}_'
                        'R{year_tplus:04d}'
                        '{month_tplus:02d}{day_tplus:02d}_HC01.nc')


# function to drop z levels when reading in data (surface only)
depth_level_index=0
def preprocess(ds):
    return ds.isel(depth=depth_level_index)

for starttime in starttimes:
    print(starttime)
    # only consider the new version (from september 2023)

    runtime = timedelta(days=31)
    endtime= starttime+runtime
    dt_file = timedelta(days=1)

    oceanfiles = create_filelist(directory_in,input_filename_base,starttime,endtime,dt_file,dt_file)


    ds = xr.open_mfdataset(oceanfiles, 
                           combine='nested', 
                           concat_dim="time",
                           preprocess= preprocess,
                           drop_variables=['so','thetao'])

    # grid resolution model
    delta_lon = ds.longitude[1]-ds.longitude[0]
    delta_lat  = ds.latitude[1]-ds.latitude[0]
    delta_t = ds.time[1]-ds.time[0]
    delta_t_s = delta_t/np.timedelta64(1, 's') # delta_t in seconds
    
    # define left points grids
    lon_left = ds.longitude - 0.5*delta_lon
    lon_left=lon_left.rename(longitude="lon_left")
    lat_left = ds.latitude - 0.5*delta_lat
    lat_left=lat_left.rename(latitude="lat_left")

    time_left = ds.time-0.5*delta_t
    time_left = time_left.rename(time='time_left')
    
    # add left points to coordinates data
    ds = ds.assign_coords(lon_left = lon_left)
    ds = ds.assign_coords(lat_left = lat_left)
    ds = ds.assign_coords(time_left = time_left)

    # calculate grid
    grid = xgcm.Grid(ds, coords={"lon": {"center": "longitude","left":"lon_left"},
                                 "lat":{"center":"latitude","left":"lat_left"},
                                 "time":{"center":"time","left":"time_left"}},
                                   autoparse_metadata=False,  periodic=False)


    # calculate derivative fields
    dudt = grid.diff(ds.uo,'time')/delta_t_s 
    dvdt = grid.diff(ds.vo,'time')/delta_t_s 
    dudx = grid.diff(ds.uo,'lon')/delta_lon
    dvdx = grid.diff(ds.vo,'lon')/delta_lon
    dudy = grid.diff(ds.uo,'lat')/delta_lat
    dvdy = grid.diff(ds.vo,'lat')/delta_lat

    ds_derivative = xr.Dataset(data_vars={'dudt':dudt,
                                          'dvdt':dvdt,
                                          'dudx':dudx,
                                          'dvdx':dvdx,
                                          'dudy':dudy,
                                          'dvdy':dvdy})

    output_path = '/storage/shared/oceanparcels/output_data/data_Meike/NWES/derivatives/'
    output_file_base = 'derivatives_field_{ys:04d}{ms:02d}{ds:04d}-{ye:04d}{me:02d}{de:04d}.nc'
    output_file = output_path + output_file_base.format(ys = starttime.year,
                                                        ms = starttime.month, 
                                                        ds = starttime.day,
                                                        ye = endtime.year,
                                                        me = endtime.month,
                                                        de = endtime.day)

    # print(output_file)
    ds_derivative.to_netcdf(output_file)
    
