"""
Calculating history term
author: Meike F. Bos
creation date: 28/10/2025
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
from particle_characteristics_functions import dynamic_viscosity_Sharqawy, factor_drag_white1991


# not sure whether this does anything

@click.command()
@click.option('--rep',default=0, help ='Particle Reynolds number',type=int)
@click.option('--pt',default='inertial_Rep_constant',help='particle type')
@click.option('--twindow',default=5,help='window time in seconds ',type=float)
@click.option('--dt_resample',default=1,help='resample timestep in seconds',type=float)

def run_experiment(pt, rep, twindow, dt_resample):
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
    tau = 3196
    nparticles = 52511
    chunck_time = 100
    coriolis = True
    gradient = True


    #general 
    av_temp_NWES = 11.983276
    av_salinity_NWES = 34.70449
    rho_water = 1027 # kg/m3 
    dynamic_viscosity_water = dynamic_viscosity_Sharqawy(av_temp_NWES,av_salinity_NWES/1000)# 1.e-3# 1.41 * 10**(-3) # kg/(ms) https://www.engineeringtoolbox.com/sea-water-properties-d_840.html (at 10 deg)
    kinematic_viscosity_water = dynamic_viscosity_water / rho_water
    diameter = 0.25 # m
    B=0.68
    cs = {'Mei':{'c1':2,'c2':0.105},
                'Kim':{'c1':2.5,'c2':0.126},
                'Dorgan':{'c1':2.5,'c2':0.2}}
    omega_earth =  7.2921e-5 #[rad/sec]



    

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
    drop_variables=['B','tau','z','flift_x','flift_y','fcor_x','fcor_y','fgradient_x','fgradient_y'],
    decode_times=False)
    ds = ds.isel(obs=slice(1,None)) # skip first step.

    dt_data = (ds.time[0,5]-ds.time[0,4]).values #time spacing data

    Nwindow = int(twindow/dt_data)+1
    Nwindow_resample = int(twindow/dt_resample)+1

    tmax = ds.time[0].max(skipna=True).values # maximum timestep in seconds
    tmin = ds.time[0].min(skipna=True).values
    nmax = int((tmax-tmin)/dt_data) # rounded maximum timestep

    
    
    def resample_time(ds, n_resample):
        # Convert obs to numpy array
        obs = ds['obs']
        obs_vals = obs.values
        # Create new obs_resampled n_resample x the original points
        obs_resampled = np.linspace(obs_vals.min(), obs_vals.max(), num=n_resample+1)
        time_resampled = ds['time'].interp(obs=('obs_resampled', obs_resampled))
        ds = ds.assign_coords(obs_resampled=obs_resampled)
        ds['time_resampled'] = time_resampled
        return ds 
    
    def resample_time_2(ds, n_resample,nmin):
        # Convert obs to numpy array
        ds_sel = ds.isel(obs=slice(nmin,None))
        obs = ds_sel['obs']
        obs_vals = obs.values
        # Create new obs_resampled n_resample x the original points
        obs_resampled = np.linspace(obs_vals.min(), obs_vals.max(), num=n_resample+1)
        time_resampled = ds['time'].interp(obs=('obs_resampled', obs_resampled))
        ds = ds.assign_coords(obs_resampled=obs_resampled)
        ds['time_resampled'] = time_resampled
        return ds 

    ds = ds.isel(obs=slice(0,nmax+1),trajectory=slice(None,None,1))#,trajectory=slice(None,None,1000)) # resample to every hour to get rid of artifial spikes in uslip # ,trajectory=slice(0,100,1)
    Uslip = np.sqrt(ds.uslip**2 + ds.vslip**2) 
    ds = ds.assign({'Uslip':Uslip})

    #settings of analysis
    n_resample = int((tmax-tmin)/dt_resample)

    Nstart = int(twindow/3600) + 1
    ds = resample_time(ds,n_resample).load()


    def cs_resample_and_derivative(v,t, tresample):
        """
        function that takes an xarray dataarray with time coordinates and returns it resampled function and derivative 
        """
        mask = ~np.isnan(v)
        if(np.sum(mask)<2):
            nan_array = np.full_like(tresample, np.nan, dtype=float)
            return nan_array, nan_array
        else:
            cs = CubicSpline(t[mask],v[mask])
            data_resampled = cs(tresample)
            data_derivative_resampled = cs.derivative()(tresample)
            return data_resampled, data_derivative_resampled

    def cs_resample(v, t, tresample):
        """
        function that takes an xarray dataarray with time coordinates and returns it resampled function
        """
        mask = ~np.isnan(v)
        if(np.sum(mask)<2):
            nan_array = np.full_like(tresample, np.nan, dtype=float)
            return nan_array, nan_array
        else:
            cs = CubicSpline(t[mask],v[mask])
            data_resampled = cs(tresample)
            return data_resampled


    def velocity_factor(uslip, vslip, der_uslip, der_vslip, lat, omega_earth):
        # "for now without factor 1/2"
        f_rotation = 2 * omega_earth * np.sin(np.pi * lat /180)
        vel_x = der_uslip - f_rotation * vslip
        vel_y = der_vslip + f_rotation * uslip
        return [vel_x, vel_y]


    def f_basset(uslip, vslip, der_uslip, der_vslip, lat, omega_earth, nu, d):
        vel_vec = velocity_factor(uslip,vslip,der_uslip,der_vslip,lat,omega_earth)
        return vel_vec[0],  vel_vec[1]

    def f_mei(t, s, uslip, vslip, der_uslip, der_vslip, lat, omega_earth, c1, c2, nu, d):
        Rep = np.sqrt(uslip**2 + vslip**2) * d / nu       
        fh = (0.75 + c2 * Rep) 
        A = nu * Rep*Rep / (fh*fh * d*d)
        term2 = (np.sqrt(np.pi) / 2 *  (A * (t - s))**(3/2))**(1.0/c1)
        factor = ((1 + term2)**(-c1))
        vel_vec = velocity_factor(uslip,vslip,der_uslip,der_vslip,lat,omega_earth)
        return factor * vel_vec[0], factor * vel_vec[1]
    
    def f_mei_rep_constant(t, s, uslip, vslip, der_uslip, der_vslip, lat, omega_earth, c1, c2, nu, d, Rep):
        fh = (0.75 + c2 * Rep) 
        A = nu * Rep*Rep / (fh*fh * d*d)
        term2 = (np.sqrt(np.pi) / 2 *  (A * (t - s))**(3/2))**(1.0/c1)
        factor = ((1 + term2)**(-c1))
        vel_vec = velocity_factor(uslip,vslip,der_uslip,der_vslip,lat,omega_earth)
        return factor * vel_vec[0], factor * vel_vec[1]

    def trapezoidal_coefficients(N,delta_t):
        """
        trapezoidal coefficients based on method by hinsberg et all 2011
        """
        G = []
        G0 = 4/3 * np.sqrt(delta_t)
        G.append(G0)
        for k in range(1,N):
            Gk = np.sqrt(delta_t)*((k + 4/3)/((k + 1)**(3/2) + (k + 3/2) * (np.sqrt(k)))
                                    + (k - 4/3)/((k-1)**(3/2)+(k-  3/2)*np.sqrt(k)))
            G.append(Gk)
        GN = np.sqrt(delta_t) * (N - 4/3) / ((N-1)**(3/2)+(N-3/2)*np.sqrt(N))
        G.append(GN)
        return G 

    da_uslip_cs, da_uslip_cs_derivative = xr.apply_ufunc(cs_resample_and_derivative,
                                                        ds.uslip, ds.time.values,ds.time_resampled.values, 
                                                        input_core_dims =[["obs"],["obs"],["obs_resampled"]],
                                                        output_core_dims=[["obs_resampled"],["obs_resampled"]],
                                                        vectorize=True,join="override")
    
   
    da_vslip_cs, da_vslip_cs_derivative = xr.apply_ufunc(cs_resample_and_derivative,
                                                        ds.vslip,ds.time.values,ds.time_resampled.values, 
                                                        input_core_dims =[["obs"],["obs"],["obs_resampled"]],
                                                        output_core_dims=[["obs_resampled"],["obs_resampled"]],
                                                        vectorize=True,join="override")
    da_lat_resampled = xr.apply_ufunc(cs_resample,
                                                        ds.lat,ds.time.values,ds.time_resampled.values, 
                                                        input_core_dims =[["obs"],["obs"],["obs_resampled"]],
                                                        output_core_dims=[["obs_resampled"]],
                                                        vectorize=True,join="override")
    ds = ds.assign({'uslip_resampled':da_uslip_cs,
                    'vslip_resampled':da_vslip_cs,
                    'der_uslip_resampled':da_uslip_cs_derivative,
                    'der_vslip_resampled':da_vslip_cs_derivative,
                    'lat_resampled':da_lat_resampled})

    
    # only sample last timesstep
    Glist =np.array(trapezoidal_coefficients(Nwindow_resample, dt_resample))

    Glist_r = Glist[::-1]
    tmax= ds.time.max().values
    
    # arrays for multiple tmax
    Hmeix_list = []
    Hmeiy_list = []
    # Hbassetx_list = []
    # Hbassety_list = []



    Nmax_array = np.arange(Nstart, int(nmax),1) 
    timelist = ds.time.isel(trajectory=1 , obs = Nmax_array).values

    for tmax_i in timelist:

        tmin = tmax_i-twindow#Nwindow_resample*dt_resample

        n_tmax_i = int(tmax_i / dt_resample)+1
        n_tmin_i = int(tmin/ dt_resample)
        shift = int(ds.time_resampled[0,0].values/dt_resample)
        slist = np.arange(n_tmin_i-1, n_tmax_i, 1) *dt_resample
        ds_time_i = ds.isel(obs_resampled =slice(n_tmax_i-shift-Glist.size, n_tmax_i-shift))
      
        if(pt in ('inertial_Rep_constant', 'inertial_SM_Rep_constant')):
            fmei_x, fmei_y = xr.apply_ufunc(f_mei_rep_constant,tmax_i, slist, 
                                        ds_time_i.uslip_resampled, ds_time_i.vslip_resampled,
                                        ds_time_i.der_uslip_resampled, ds_time_i.der_vslip_resampled, 
                                        ds_time_i.lat_resampled, omega_earth,
                                        cs['Kim']['c1'], cs['Kim']['c2'], 
                                        kinematic_viscosity_water,diameter , Rep,
                                        input_core_dims =[[],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],[],[],[],[],[],[]],
                                                            output_core_dims=[["obs_resampled"],["obs_resampled"]],
                                                            vectorize=True)  
        else:
            fmei_x, fmei_y = xr.apply_ufunc(f_mei,tmax_i, slist, 
                                        ds_time_i.uslip_resampled, ds_time_i.vslip_resampled,
                                        ds_time_i.der_uslip_resampled, ds_time_i.der_vslip_resampled, 
                                        ds_time_i.lat_resampled, omega_earth,
                                        cs['Kim']['c1'], cs['Kim']['c2'], 
                                        kinematic_viscosity_water,diameter ,
                                        input_core_dims =[[],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],[],[],[],[],[]],
                                                            output_core_dims=[["obs_resampled"],["obs_resampled"]],
                                                            vectorize=True)





        # fbasset_x, fbasset_y = xr.apply_ufunc(f_basset,
        #                             ds_time_i.uslip_resampled, ds_time_i.vslip_resampled,
        #                             ds_time_i.der_uslip_resampled, ds_time_i.der_vslip_resampled, 
        #                             ds_time_i.lat_resampled, omega_earth,
        #                             kinematic_viscosity_water,diameter ,
        #                             input_core_dims =[["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],["obs_resampled"],[],[],[]],
        #                                                 output_core_dims=[["obs_resampled"],["obs_resampled"]],
        #                                                 vectorize=True)#,join="override")
        
        Hmeix, Hmeiy = [(Glist_r * fmei_x).sum(dim='obs_resampled').values,(Glist_r * fmei_y).sum(dim='obs_resampled').values]
        # Hbassetx, Hbassety = [(Glist_r * fbasset_x).sum(dim='obs_resampled').values,(Glist_r * fbasset_y).sum(dim='obs_resampled').values]

        Hmeix_list.append(Hmeix)
        Hmeiy_list.append(Hmeiy)
        # Hbassetx_list.append(Hbassetx)
        # Hbassety_list.append(Hbassety)
    ds_history = xr.Dataset(
        data_vars=dict(
            History_Mei_x = (["time","trajectory"],Hmeix_list),
            History_Mei_y = (["time","trajectory"],Hmeiy_list),

            # History_Basset_x = (["time","trajectory"],Hbassetx_list),
            # History_Basset_y = (["time","trajectory"],Hbassety_list),
            # uslip = (["trajectory","time"],ds.uslip.isel(obs=slice(Nstart, ds.obs[-1].values)).values),
            # vslip = (["trajectory","time"],ds.vslip.isel(obs=slice(Nstart, ds.obs[-1].values)).values)

        ),
        coords=dict(
            time = timelist,
            trajectory = np.arange(0,int(nparticles),1)
        ),
        attrs=dict(description  ="History term calculated", 
                    particle_type = pt,
                    Rep = Rep,
                    # starttime = starttime,
                    twindow = twindow, 
                    dt_resample =dt_resample, 
                    Nresample = n_resample),
        )

    twindow_int = int(twindow*10) 
    dt_resample_int=int(dt_resample*10) 
    if(pt in ('inertial_Rep_constant', 'inertial_SM_Rep_constant')):
        output_name = f'/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/NWES/{pt}/history_term_Mei_Rep{Rep:04d}_twindow{twindow_int:06d}_tresample{dt_resample_int:04d}.netcdf'
    else: 
        output_name = f'/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/NWES/{pt}/history_term_Mei_twindow{twindow_int:06d}_tresample{dt_resample_int:04d}.netcdf'
    ds_history.to_netcdf(output_name)
    print(output_name + ' simulation finished')

if __name__ == '__main__':
    run_experiment()