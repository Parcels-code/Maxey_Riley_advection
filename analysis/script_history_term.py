"""
Calculating history term
author: Meike F. Bos
creation date: 31/07/2025
description: script to calcualte create history term per particle
"""

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline # maybe not needed
import scipy.special as sc

import sys
sys.path.append("/nethome/4291387/Maxey_Riley_advection/Maxey_Riley_advection/src")
from particle_characteristics_functions import stokes_relaxation_time, Re_particle, factor_drag_white1991, diffusion_time, slip_force
from history_term_functions import Basset_kernel, Mei1992_kernel, history_timescale, Daitche, Hinsberg, f_Mei1992, History_Force_Hinsberg_Mei_kernel, History_Force_Hinsberg_Basset_kernel
from analysis_functions_xr import derivative_backward, calc_tidal_av
from analysis_functions import make_PDF, make_lognormal_PDF

# not sure whether this does anything
import dask.array as daskarray
from numba import njit


# settings of data
starttime = datetime(2023,9,1,0,0,0)
runtime = timedelta(days =2)
endtime = starttime + runtime
loc = 'NWES'
land_handling = 'anti_beaching'
coriolis = True
B = 0.68
tau = 2994.76
Replist = [0,10,100,450,1000]
starttimes = [datetime(2023, 9, 1, 0, 0, 0, 0)]
nparticles = 52511
chunck_time = 100
coriolis = True
gradient = True
dt = timedelta(minutes=5).seconds


# import data set names and directories
base_directory = '/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/NWES/test_history_term/'
basefile_Rep_constant = (base_directory + '{particle_type}/{loc}_'
                 'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                 'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_'
                 '_Rep_{Rep:04d}_B{B:04d}_tau{tau:04d}_{land_handling}_cor_{coriolis}_gradient_{gradient}.zarr')

basefile_Rep_drag = (base_directory + '{particle_type}/{loc}_start{y_s:04d}_{m_s:02d}_{d_s:02d}'
                 '_end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_B{B:04d}_tau{tau:04d}_{land_handling}_cor_{coriolis}_gradient_{gradient}.zarr')

basefiles={
           'inertial_Rep_constant':basefile_Rep_constant,
           'inertial_SM_Rep_constant':basefile_Rep_constant, 
           'inertial_SM_drag_Rep':basefile_Rep_drag,
           'inertial_drag_Rep':basefile_Rep_drag}

particle_types = ['inertial_drag_Rep','inertial_Rep_constant']#,'inertial_Rep_constant'] # 
names = {'inertial_Rep_constant':'MR',
         'inertial_SM_Rep_constant':'SM MR', 
         'inertial_SM_drag_Rep':'SM MR flexible Re$_p$',
         'inertial_drag_Rep':'MR flexible Re$_p$'}
Rep_dict = {'inertial_Rep_constant':Replist,
         'inertial_SM_Rep_constant':Replist,
         'inertial_SM_drag_Rep':[None],
         'inertial_drag_Rep':[None]}

# import data
# create data directory
data = {}
for pt in particle_types:
    data[pt]={}


for pt in particle_types:
    for Rep in Rep_dict[pt]:
        file = basefiles[pt].format(loc=loc,
                                   y_s=starttime.year,
                                   m_s=starttime.month,
                                   d_s=starttime.day,
                                   y_e=endtime.year,
                                   m_e=endtime.month,
                                   d_e=endtime.day,
                                   B = int(B * 1000), 
                                   tau = int(tau ),
                                   land_handling = land_handling, 
                                   coriolis = coriolis,
                                   gradient = gradient,
                                   particle_type = pt,
                                   Rep = Rep)
        ds = xr.open_dataset(file,
                             engine='zarr',
                             chunks={'trajectory':nparticles, 'obs':chunck_time},
                             drop_variables=['B','tau','z'],
                             decode_times=False) 

        Uslip = np.sqrt(ds.uslip**2 + ds.vslip**2) 
        ds = ds.assign({'Uslip':Uslip})
        data[pt][Rep]= ds 

def resample_time(ds, n_resample, obs_vals):
    # Convert obs to numpy array
    obs = ds['obs']
    obs_vals = obs.values
    # Create new obs_resampled n_resample x the original points
    obs_resampled = np.linspace(obs_vals.min(), obs_vals.max(), num=(len(obs_vals) - 1) * n_resample + 1)
    time_resampled = ds['time'].interp(obs=('obs_resampled', obs_resampled))
    ds = ds.assign_coords(obs_resampled=obs_resampled)
    ds['time_resampled'] = time_resampled
    return ds 

def take_spine(v, t, tresample):
    """
    function that takes an xarray dataarray with time coordinates and returns it resampled function and derivative 
    """

    cs = CubicSpline(t, v)
    data_resampled = cs(tresample)
    data_derivative_resampled = cs.derivative()(tresample)
    return data_resampled, data_derivative_resampled

ds_select = data['inertial_Rep_constant'][100].isel(trajectory=1)


def trapezoidal_coefficients(N,delta_t):
    """
    trapezoidal coefficients based on method by hinsberg et all 2011
    """
    G = []
    G0 = 4/3 * np.sqrt(delta_t)
    G.append(G0)
    for k in range(1,N):
        Gk = np.sqrt(delta_t)*((k+4/3)/((k+1)**(3/2)+(k+3/2) * (np.sqrt(k)))
                                + (k-4/3)/((k-1)**(3/2)+(k-3/2)*np.sqrt(k)))
        G.append(Gk)
    GN = np.sqrt(delta_t) * (N - 4/3) * ((N-1)**(3/2)+(N-3/2)*np.sqrt(N))
    G.append(GN)
    return G 
    

print(ds)
da_time = ds_select.time
print(da_time)
nresample =10
# print(da_time.obs.size)
resample_freq = np.arange(0,da_time.obs.size,1/nresample)
# print(resample_freq.size)
da_time_resample = da_time.interp(obs=resample_freq)
print(da_time_resample.values)
# da_test = xr.apply_ufunc(take_spline, ds_select.uslip, da_time, )
# ds_select.uslip.
# define functions 
# def take_spline(v, t):
#     cs = CubicSpline(t, v)
#     return cs(t)

# # functions to calculate history term
# # Define a function that works on 1D velocity arrays
# def spline_derivative(v, t):
#     cs = CubicSpline(t, v)
#     return cs.derivative()(t)

# def spline(v, t):
#     cs = CubicSpline(t, v)
#     return cs(t)

# def make_windowed_da(da: xr.DataArray, nwindow: int) -> xr.DataArray:
#     traj_dim = 'trajectory'
#     obs_dim = 'obs'

#     # Pad with zeros on the left (before obs=0)
#     padded = np.pad(da.values, ((0, 0), (nwindow - 1, 0)), mode='constant')

#     # Create windowed array
#     result = np.lib.stride_tricks.sliding_window_view(padded, window_shape=nwindow, axis=1)

#     # result shape: (trajectory, obs, nwindow)
#     coords = {
#         'trajectory': da.coords[traj_dim],
#         'obs': da.coords[obs_dim],
#         'nwindow': np.arange(nwindow)
#     }

#     return xr.DataArray(result, dims=(traj_dim, obs_dim, 'nwindow'), coords=coords)




# def make_windowed_da_dask(da_xr: xr.DataArray, nwindow: int) -> xr.DataArray:
#     traj_dim = 'trajectory'
#     obs_dim = 'obs'
    
#     darr = da_xr.data  # this is a dask array
    
#     # Pad along obs axis (axis=1)
#     pad_width = ((0, 0), (nwindow - 1, 0))  # pad only obs axis on left
#     padded = daskarray.pad(darr, pad_width, mode='constant', constant_values=0)
    
#     # Use dask's sliding_window_view
#     result = daskarray.lib.stride_tricks.sliding_window_view(padded, window_shape=nwindow, axis=1)
    
#     # result shape: (trajectory, obs, nwindow)
#     coords = {
#         traj_dim: da_xr.coords[traj_dim],
#         obs_dim: da_xr.coords[obs_dim],
#         'nwindow': np.arange(nwindow)
#     }
    
#     return xr.DataArray(result, dims=(traj_dim, obs_dim, 'nwindow'), coords=coords)


# def f_Mei1992_unit_vectorized(dudt, t, c1, c2, Rep, h, tau_diff):
#     """
#     dudt: array of shape (nwindow,)
#     t: scalar
#     Rep: scalar or 1D if needed
#     Returns: array of shape (nwindow,)
#     """
#     n = np.arange(len(dudt))
#     delta_t = t - h * n

#     kernel = (
#         ((delta_t**3 * 12 * np.pi) / (tau_diff * t**3)) ** 0.5
#         * Rep**3 / (16 * np.pi * (0.75 + c2 * Rep)**3)
#     )
#     factor = (1 + kernel ** (1 / c1)) ** (-c1)

#     return dudt * factor

