# import needed packages
import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt
import xgcm
import cartopy.crs as ccrs 
import cartopy as cart
import cartopy.feature as cfeature
from datetime import datetime, timedelta

starttime = datetime(2023,9,1, 0, 0, 0, 0)
endtime = datetime(2023,10,1,0,0,0)
dt_name_field = timedelta(days=1)
dt_field = timedelta(hours=1)

# for fieldsets
field_directory = ('/storage/shared/oceanparcels/input_data/CopernicusMarineService/'
                    'NORTHWESTSHELF_ANALYSIS_FORECAST_PHY_004_013/')


input_filename = ('CMEMS_v6r1_NWS_PHY_NRT_NL_01hav3D_'
                '{year_t:04d}{month_t:02d}{day_t:02d}_'
                '{year_t:04d}{month_t:02d}{day_t:02d}_'
                'R{year_tplus:04d}{month_tplus:02d}{day_tplus:02d}_HC01.nc')
# define function to make list of files
def create_filelist(input_directory, str ,time_start,time_end,dt_field):
    time = time_start
    files = []
    while (time<=time_end):
        time_tplus = time+dt_field
        y_t =time.year
        m_t = time.month
        d_t = time.day
        y_tp = time_tplus.year
        m_tp=time_tplus.month
        d_tp=time_tplus.day
        files.append(input_directory+str.format(year_t = y_t, month_t = m_t, day_t = d_t, year_tplus = y_tp, month_tplus = m_tp, day_tplus =d_tp))
        time += dt_field
        
    return files


#
depth_level_index=0

def preprocess(ds):
    return ds.isel(depth=depth_level_index)

oceanfiles=create_filelist(field_directory, input_filename,
                                starttime, endtime, dt_name_field)

depth_level_index=0

def preprocess(ds):
    return ds.isel(depth=depth_level_index)

ds = xr.open_mfdataset(oceanfiles, combine='nested', concat_dim="time",preprocess= preprocess)#,drop_variables=['so','thetao'])

# use xarray diff function to calculate derivatives 
dlon = (ds['longitude'][1]-ds['longitude'][0]).values
dlat = (ds['latitude'][1]-ds['latitude'][0]).values
dtime = (ds['time'][1]-ds['time'][0]).values


time, lon, lat = (ds['time'].values, ds['longitude'].values,ds['latitude'].values)
LON, LAT = np.meshgrid(lon,lat)
LON_dx, LAT_dx = np.meshgrid(lon[0:-1]+0.5*dlon,lat)
LON_dy, LAT_dy = np.meshgrid(lon,lat[0:-1]+0.5*dlat)

Rearth = 6371 * 10**3 # in k
Omega_earth =7.2921 * (10**-5) # angular velocity earth in rad/s
deg2rad = np.pi / 180.


# define jacobean 
Jtime = 1/3600 # seconds per hour
Jx = 1 / (2 * Rearth * np.arcsin(np.sqrt(0.5 * ( np.cos(LAT[:,1:] * deg2rad)**2) * (1 - np.cos(dlon * deg2rad)))))
Jy = 1 / (Rearth * dlat * deg2rad)

# dimensions (time, lat, lon)
dvdt = ds['vo'].diff(dim='time',n=1,label="upper") * Jtime
dvdx = ds['vo'].diff(dim='longitude',n=1,label="upper") * Jx
dvdy = ds['vo'].diff(dim='latitude',n=1,label="upper") * Jy

dudt = ds['uo'].diff(dim='time',n=1,label="upper") * Jtime
dudx = ds['uo'].diff(dim='longitude',n=1,label="upper") * Jx
dudy = ds['uo'].diff(dim='latitude',n=1,label="upper") * Jy


# 
dvdx['longitude']=dvdx['longitude']-0.5*dlon
dudx['longitude']=dudx['longitude']-0.5*dlon
dvdy['latitude']=dvdy['latitude']-0.5*dlat
dvdy['latitude']=dvdy['latitude']-0.5*dlat
dvdt['time']=dvdt['time']-0.5*dtime
dudt['time']=dudt['time']-0.5*dtime

# calculate u and v component material derivative field
dvdx_agrid=dvdx.interp(longitude=lon[1:-1])[1:-1,1:-1,:]
dudx_agrid=dudx.interp(longitude=lon[1:-1])[1:-1,1:-1,:]
dvdy_agrid=dvdy.interp(latitude=lat[1:-1])[1:-1,:,1:-1]
dudy_agrid=dudy.interp(latitude=lat[1:-1])[1:-1,:,1:-1]

dvdt_agrid = dvdt.interp(time=time[1:-1])[:,1:-1,1:-1]
dudt_agrid = dudt.interp(time=time[1:-1])[:,1:-1,1:-1]
u = ds['uo'][1:-1,1:-1,1:-1]
v = ds['vo'][1:-1,1:-1,1:-1]
DuDt = (dudt_agrid + u * dudx_agrid + v * dudy_agrid)#.load()
DvDt = (dvdt_agrid + u * dvdx_agrid + v * dvdy_agrid)#.load()
vorticity = dvdx_agrid - dudy_agrid
mean_vorticity = np.abs(vorticity).mean().values

gradient_speed = np.sqrt(DuDt**2 + DvDt**2)
speed = np.sqrt(u**2+v**2)
mean_gradient = gradient_speed.mean().values
mean_speed = speed.mean().values
print(mean_gradient)
print(mean_speed)
print(mean_vorticity)
# print(mean_gradient/mean_spe)

