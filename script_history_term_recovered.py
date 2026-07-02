"""
Calculating history term
author: Meike F. Bos
creation date: 31/07/2025
description: script to calcualte create history term per particle
"""
import click 
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline # maybe not needed
import scipy.special as sc
import time



import sys
sys.path.append("/nethome/4291387/Maxey_Riley_advection/Maxey_Riley_advection/src")
from particle_characteristics_functions import stokes_relaxation_time, Re_particle, factor_drag_white1991, diffusion_time, slip_force
from history_term_functions import Basset_kernel, Mei1992_kernel, history_timescale, Daitche, Hinsberg, f_Mei1992, History_Force_Hinsberg_Mei_kernel, History_Force_Hinsberg_Basset_kernel
from analysis_functions_xr import derivative_backward, calc_tidal_av
from analysis_functions import make_PDF, make_lognormal_PDF

# not sure whether this does anything
import dask.array as daskarray
from numba import njit

@click.command()
@click.option('--rep',default=0, help ='Particle Reynolds number')
@click.option('--pt',default='inertial_Rep_constant',help='particle type')
@click.option('--nwindow',default=12,help='number of (original) timesteps taken as window')
@click.option('--nresample',default=10,help='number of extra points added between 2 original timesteps')

def run_experiment(pt, rep, nresample, nwindow):
    start = time.process_time()
    Rep = rep

    # settings of data
    starttime = datetime(2023,9,1,0,0,0)
    runtime = timedelta(days =2)
    endtime = starttime + runtime
    loc = 'NWES'
    land_handling = 'anti_beaching'
    coriolis = True
    B = 0.68
    tau = 3196#  2994.76
    Replist = [0,10,100,450,1000]
    starttimes = [datetime(2023, 9, 1, 0, 0, 0, 0)]
    nparticles = 52511
    chunck_time = 100
    coriolis = True
    gradient = True
    dt = timedelta(minutes=5).seconds
    Tmax = 12 * 48

    #general 
    rho_water = 1027 # kg/m3 https://www.engineeringtoolbox.com/sea-water-properties-d_840.html (at 10 deg)
    dynamic_viscosity_water = 1.41 * 10**(-3) # kg/(ms) https://www.engineeringtoolbox.com/sea-water-properties-d_840.html (at 10 deg)
    kinematic_viscosity_water = dynamic_viscosity_water / rho_water
    diameter =0.25 # m
    B=0.68
    cs = {'Mei':{'c1':2,'c2':0.105},
                'Kim':{'c1':2.5,'c2':0.126},
                'Dorgan':{'c1':2.5,'c2':0.2}}
    omega_earth =  7.2921e-5 #[rad/sec]


    #settings of analysis
    Nwindow = nwindow
    n_resample = nresample
    dt_resample = dt /n_resample

    # import data set names and directories
    base_directory = '/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/NWES/'
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


    def resample_time(ds, n_resample):
        # Convert obs to numpy array
        obs = ds['obs']
        obs_vals = obs.values
        # Create new obs_resampled n_resample x the original points
        obs_resampled = np.linspace(obs_vals.min(), obs_vals.max(), num=(len(obs_vals) - 1) * n_resample + 1)
        time_resampled = ds['time'].interp(obs=('obs_resampled', obs_resampled))
        ds = ds.assign_coords(obs_resampled=obs_resampled)
        ds['time_resampled'] = time_resampled
        return ds 

    def cs_resample_and_derivative(v, t, tresample):
        """
        function that takes an xarray dataarray with time coordinates and returns it resampled function and derivative 
        """
        mask = ~np.isnan(v)
        cs = CubicSpline(t[mask],v[mask])
        data_resampled = cs(tresample)
        data_derivative_resampled = cs.derivative()(tresample)
        return data_resampled, data_derivative_resampled

    def cs_resample(v, t, tresample):
        """
        function that takes an xarray dataarray with time coordinates and returns it resampled function
        """
        mask = ~np.isnan(v)
        cs = CubicSpline(t[mask],v[mask])
        data_resampled = cs(tresample)
        return data_resampled


    def velocity_factor(uslip, vslip, der_uslip, der_vslip, lat, omega_earth):
        # "for now without factor 1/2"
        f_rotation = 1 * omega_earth * np.sin(np.pi * lat /180)
        vel_x = der_uslip - f_rotation * vslip
        vel_y = der_vslip + f_rotation * uslip
        return [vel_x, vel_y]


    def f_basset(uslip, vslip, der_uslip, der_vslip, lat, omega_earth, nu, d):
        factor =  (d**2 / (4 * np.pi * nu) )**(1/2)

        vel_vec = velocity_factor(uslip,vslip,der_uslip,der_vslip,lat,omega_earth)
        return factor * vel_vec[0], factor * vel_vec[1]

    def f_mei(t, s, uslip, vslip, der_uslip, der_vslip, lat, omega_earth, c1, c2, nu, d):
        Rep = np.sqrt(uslip**2 + vslip**2) * d / nu
        # Rep = np.mean(Rep).values
        # Rep = 0 # for testing
        
        A = (4 * np.pi * nu / (d*d))**(1/(2*c1))
        fh = (0.75 + c2*Rep)**3
        B = (np.pi * nu * nu * (t - s )**(3/2) /(fh * d**4) * Rep**3)**(1/c1)
        factor = (A + B)**(-c1)
        vel_vec = velocity_factor(uslip,vslip,der_uslip,der_vslip,lat,omega_earth)
        return factor * vel_vec[0], factor * vel_vec[1]
    
    def f_mei_rep_constant(t, s, uslip, vslip, der_uslip, der_vslip, lat, omega_earth, c1, c2, nu, d, Rep):
        # Rep = np.sqrt(uslip**2 + vslip**2) * d / nu
        # Rep = np.mean(Rep).values
        # Rep = 0 # for testing
        
        A = (4 * np.pi * nu / (d*d))**(1/(2*c1))
        fh = (0.75 + c2*Rep)**3
        B = (np.pi * nu * nu * (t - s )**(3/2) /(fh * d**4) * Rep**3)**(1/c1)
        factor = (A + B)**(-c1)
        vel_vec = velocity_factor(uslip,vslip,der_uslip,der_vslip,lat,omega_earth)
        return factor * vel_vec[0], factor * vel_vec[1]

    def f_talaei_rep_constant(t, s, uslip, vslip, der_uslip, der_vslip, lat, omega_earth, nu, d, Rep):
        """ f(s) so velocity lists are indexed by s"""
        cs = factor_drag_white1991(Rep)
        A = (4 * np.pi * nu / (d*d))**(1/2)
        B = np.sqrt(4 * np.pi * (t - s ))**(1/2) /(d**2 ) * Rep
        factor = (A + B)**(-1)
        vel_vec = velocity_factor(uslip,vslip,der_uslip,der_vslip,lat,omega_earth)
        return cs * factor * vel_vec[0],cs * factor * vel_vec[1]
    
    def f_talaei(t, s, uslip, vslip, der_uslip, der_vslip, lat, omega_earth, nu, d):
        """ f(s) so velocity lists are indexed by s"""
        
        Rep = np.sqrt(uslip**2 + vslip**2) * d / nu
        # Rep = 0 # for testing
        cs = factor_drag_white1991(Rep)
        A = (4 * np.pi * nu / (d*d))**(1/2)
        B = np.sqrt(4 * np.pi * (t - s ))**(1/2) /(d**2 ) * Rep
        factor = (A + B)**(-1)
        vel_vec = velocity_factor(uslip,vslip,der_uslip,der_vslip,lat,omega_earth)
        return cs * factor * vel_vec[0],cs * factor * vel_vec[1]


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
        GN = np.sqrt(delta_t) * (N - 4/3) / ((N-1)**(3/2)+(N-3/2)*np.sqrt(N))
        G.append(GN)
        return G 


        
        

    # testcase
    ds = ds.isel(obs = slice(0,Tmax)).load()
    ds = resample_time(ds, n_resample)

    da_uslip_cs, da_uslip_cs_derivative = xr.apply_ufunc(cs_resample_and_derivative,
                                                        ds.uslip,dt * ds.obs.values,dt_resample *  ds.obs_resampled.values, 
                                                        input_core_dims =[["obs"],["obs"],["obs_resampled"]],
                                                        output_core_dims=[["obs_resampled"],["obs_resampled"]],
                                                        vectorize=True,join="override")
    da_vslip_cs, da_vslip_cs_derivative = xr.apply_ufunc(cs_resample_and_derivative,
                                                        ds.vslip,dt * ds.obs.values,dt_resample *  ds.obs_resampled.values, 
                                                        input_core_dims =[["obs"],["obs"],["obs_resampled"]],
                                                        output_core_dims=[["obs_resampled"],["obs_resampled"]],
                                                        vectorize=True,join="override")
    da_lat_resampled = xr.apply_ufunc(cs_resample,
                                                        ds.lat,dt * ds.obs.values,dt_resample *  ds.obs_resampled.values, 
                                                        input_core_dims =[["obs"],["obs"],["obs_resampled"]],
                                                        output_core_dims=[["obs_resampled"]],
                                                        vectorize=True,join="override")
    ds = ds.assign({'uslip_resampled':da_uslip_cs,
                    'vslip_resampled':da_vslip_cs,
                    'der_uslip_resampled':da_uslip_cs_derivative,
                    'der_vslip_resampled':da_vslip_cs_derivative,
                    'lat_resampled':da_lat_resampled})

    # Nwindow = 60 # 5 hours in in original timesteps of 5 min
    Nwindow_resampled = Nwindow * n_resample
    Nstart = Nwindow + 12 #  2 * int(Nwindow/n_resample) +1
    i_array = np.arange(Nstart, ds.obs[-1].values,1)
    Hmeix_list = []
    Hmeiy_list = []
    # Htalaeix_list = []
    # Htalaeiy_list = []
    # Hbassetx_list = []
    # Hbassety_list = []
    Glist =np.array(trapezoidal_coefficients(Nwindow_resampled,dt_resample))
    Glist_r = Glist[::-1]
    slist = np.arange(0,Nwindow_resampled+1,1)*dt_resample
    # timelist = []
    timelist = ds.time.isel(trajectory=1 , obs = slice(Nstart, ds.obs[-1].values)).values
    # print(timelist)
    # print(Nstart)
    # print(ds.obs[-1].values)
    for i in i_array:
        time_i = ds.time.isel(obs = i)
        # tvalue = time_i.isel(trajectory=1).values
        ds_time_i = ds.isel(obs_resampled =slice(i*n_resample-Nwindow_resampled-1,i*n_resample))

        if(pt in ('inertial_Rep_constant', 'inertial_SM_Rep_constant')):
            fmei_x, fmei_y = xr.apply_ufunc(f_mei,time_i, slist, 
                                        ds_time_i.uslip_resampled, ds_time_i.vslip_resampled,
                                        ds_time_i.der_uslip_resampled, ds_time_i.der_vslip_resampled, 
                                        ds_time_i.lat_resampled, omega_earth,
                                        cs['Kim']['c1'], cs['Kim']['c2'], 
                                        kinematic_viscosity_water,diameter , Rep,
                                        input_core_dims =[[],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],[],[],[],[],[],[]],
                                                            output_core_dims=[["obs_resampled"],["obs_resampled"]],
                                                            vectorize=True)#,join="override")
            # ftalaei_x, ftalaei_y = xr.apply_ufunc(f_talaei_rep_constant,time_i, slist, 
            #                         ds_time_i.uslip_resampled, ds_time_i.vslip_resampled,
            #                         ds_time_i.der_uslip_resampled, ds_time_i.der_vslip_resampled, 
            #                         ds_time_i.lat_resampled, omega_earth,
            #                         kinematic_viscosity_water,diameter , Rep,
            #                         input_core_dims =[[],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],[],[],[],[]],
            #                                             output_core_dims=[["obs_resampled"],["obs_resampled"]],
            #                                             vectorize=True)#,join="override")

        
        else:
            fmei_x, fmei_y = xr.apply_ufunc(f_mei,time_i, slist, 
                                        ds_time_i.uslip_resampled, ds_time_i.vslip_resampled,
                                        ds_time_i.der_uslip_resampled, ds_time_i.der_vslip_resampled, 
                                        ds_time_i.lat_resampled, omega_earth,
                                        cs['Kim']['c1'], cs['Kim']['c2'], 
                                        kinematic_viscosity_water,diameter ,
                                        input_core_dims =[[],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],[],[],[],[],[]],
                                                            output_core_dims=[["obs_resampled"],["obs_resampled"]],
                                                            vectorize=True)#,join="override")


            # ftalaei_x, ftalaei_y = xr.apply_ufunc(f_talaei,time_i, slist, 
            #                             ds_time_i.uslip_resampled, ds_time_i.vslip_resampled,
            #                             ds_time_i.der_uslip_resampled, ds_time_i.der_vslip_resampled, 
            #                             ds_time_i.lat_resampled, omega_earth,
            #                             kinematic_viscosity_water,diameter ,
            #                             input_core_dims =[[],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],[],[],[]],
            #                                                 output_core_dims=[["obs_resampled"],["obs_resampled"]],
            #                                                 vectorize=True)#,join="override")


        # fbasset_x, fbasset_y = xr.apply_ufunc(f_basset,
        #                             ds_time_i.uslip_resampled, ds_time_i.vslip_resampled,
        #                             ds_time_i.der_uslip_resampled, ds_time_i.der_vslip_resampled, 
        #                             ds_time_i.lat_resampled, omega_earth,
        #                             kinematic_viscosity_water,diameter ,
        #                             input_core_dims =[["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],[],[],[]],
        #                                                 output_core_dims=[["obs_resampled"],["obs_resampled"]],
        #                                                 vectorize=True)#,join="override")

        Hmeix, Hmeiy = [(Glist_r * fmei_x).sum(dim='obs_resampled').values,(Glist_r * fmei_y).sum(dim='obs_resampled').values]
        # Htalaeix, Htalaeiy = [(Glist_r * ftalaei_x).sum(dim='obs_resampled').values,(Glist_r * ftalaei_y).sum(dim='obs_resampled').values]
        # Hbassetx, Hbassety = [(Glist_r * fbasset_x).sum(dim='obs_resampled').values,(Glist_r * fbasset_y).sum(dim='obs_resampled').values]
        Hmeix_list.append(Hmeix)
        Hmeiy_list.append(Hmeiy)
        # Htalaeix_list.append(Htalaeix)
        # Htalaeiy_list.append(Htalaeiy)
        # Hbassetx_list.append(Hbassetx)
        # Hbassety_list.append(Hbassety)


    # put history terms in xarray dataset and write this dataset to file
    # print(timelist)
    # print(timelist)
    # timelist = np.asarray(timelist)
    # # timelist = timelist.flatten()
    # print(timelist)
    ds_history = xr.Dataset(
        data_vars=dict(
            History_Mei_x = (["time","trajectory"],Hmeix_list),
            History_Mei_y = (["time","trajectory"],Hmeiy_list),
            # History_Talaei_x = (["time","trajectory"],Htalaeix_list),
            # History_Talaei_y = (["time","trajectory"],Htalaeiy_list),
            # History_Basset_x = (["time","trajectory"],Hbassetx_list),
            # History_Basset_y = (["time","trajectory"],Hbassety_list),
            uslip = (["trajectory","time"],ds.uslip.isel(obs=slice(Nstart, ds.obs[-1].values)).values),
            vslip = (["trajectory","time"],ds.vslip.isel(obs=slice(Nstart, ds.obs[-1].values)).values)

        ),
        coords=dict(
            time = timelist,
            trajectory = np.arange(0,nparticles,1)
        ),
        attrs=dict(description  ="History term calculated", 
                    particle_type = pt,
                    Rep = Rep,
                    # starttime = starttime,
                    Nwindow = Nwindow,
                    Nresample = n_resample,
                    Nstart = Nstart),
        )
    if(pt in ('inertial_Rep_constant', 'inertial_SM_Rep_constant')):
        output_name = f'/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/NWES/{pt}/history_term_Rep{Rep:04d}_Nwindow{nwindow:03d}_Nresample{nresample:03d}.netcdf'
    else: 
        output_name = f'/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/NWES/{pt}/history_term_Nwindow{nwindow:03d}_Nresample{nresample:03d}.netcdf'
    ds_history.to_netcdf(output_name)
    print(f'simulation time: {time.process_time() - start}')


if __name__ == '__main__':
    run_experiment()