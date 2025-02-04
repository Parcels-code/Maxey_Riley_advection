"""
Date: November 2024
author: Meike Bos <m.f.bos@uu.nl>
Description: Testing different initial velocity conditions for inertial stokes 
drifters. We simulate the stokes drifters for a few timesteps and save their 
position every integration timestep to study the difference between (a) the slow
manifold maxeyRiley (MRSM) equation with (b) the full MR equation initialized 
with the fluid velocity or (c) the velocity calculated using the MRSM at t0. 
"""

# import needed packages
import numpy as np
import xarray as xr
from parcels import FieldSet, ParticleSet, ParticleFile, Variable
from parcels import AdvectionRK4
from datetime import datetime, timedelta
from helper import create_filelist, set_particles_region, displace_coordinates
from kernels import InertialParticle2D, deleteParticle
from kernels import InitializeParticles2D, InitializeParticles2D_MRSM
from kernels import MRAdvectionRK4_2D, MRSMAdvectionRK4_2D
from kernels import too_close_to_edge, remove_at_bounds


# set directories
field_directory = ('/storage/shared/oceanparcels/input_data/CMEMS/'
                   'NORTHWESTSHELF_ANALYSIS_FORECAST_PHY_004_013/')
land_directory = ('/storage/shared/oceanparcels/'
                  'output_data/data_Meike/NWES/')
output_directory = ('/storage/shared/oceanparcels/'
                    'output_data/data_Meike/MR_advection/NWES/test_init_conditions/')

output_file_b = (output_directory + '{particle_type}_{loc}_'
                 'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                 'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_'
                 'B{B:04d}_tau{tau:04d}.zarr')

##################################
#       Simulation settings      #
##################################

# options are tracer, tracer_random, inertial (MR) or inertial_SM (MR slow manifold), inertial_initSM (MR velocity initialized using the MR slow manifold eq)
particle_type = 'inertial_initSM'

starttime = datetime(2023, 9, 1, 0, 0, 0, 0)
release_times = np.array([  datetime(2023, 9, 1, 0, 0, 0, 0) ])
runtime = timedelta(hours=3)
endtime = release_times[-1]+runtime+timedelta(days=1)
dt_timestep = timedelta(seconds=10)
dt_write = timedelta(minutes=1)
B = 0.68
# stokes relaxation time
tau = 2759.97
loc = 'custom'
grid = 'square'

# set custom region:
startlon_release = -10
endlon_release = 0
startlat_release = 47
endlat_release = 52

indices = {}


#########################
#       Set Fields      #
#########################
start_new_dataset = datetime(2023, 9, 1, 0, 0, 0, 0)
dt_field = timedelta(days=1)
# variables and dimensions (3D)
variables = {'U': 'uo',
             'V': 'vo'}

dimensions = {'lat': 'lat',
              'lon': 'lon',
              'time': 'time'}

if (starttime >= start_new_dataset): 
    print('use new dataset (Nologin Spain)')
    dimensions = {'lat': 'latitude',
                  'lon': 'longitude',
                  'time': 'time'}
else:
    print('use old dataset (UK Met Office)')

if (starttime >= start_new_dataset):
    dt_name_field = timedelta(days=1)
    input_filename = ('CMEMS_v6r1_NWS_PHY_NRT_NL_01hav3D_'
                      '{year_t:04d}{month_t:02d}{day_t:02d}_'
                      '{year_t:04d}{month_t:02d}{day_t:02d}_'
                      'R{year_tplus:04d}{month_tplus:02d}{day_tplus:02d}_HC01.nc')
else:
    dt_name_field = timedelta(days=2)
    input_filename = ('metoffice_foam1_amm15_NWS_CUR_'
                      'b{year_t:04d}{month_tr:02d}{day_t:02d}_'
                      'hi{year_tplus:04d}{month_tplus:02d}{day_tplus:02d}.nc')
    
oceanfiles=create_filelist(field_directory, input_filename,
                               starttime, endtime, dt_field, dt_name_field)

# fieldset interp method

fieldset = FieldSet.from_netcdf(oceanfiles, variables, dimensions, indices=indices,
                                        allow_time_extrapolation="False")#
fieldset.interp_method = 'free_slip'

# angular velocity earth in radians/second
fieldset.add_constant('Omega_earth', 7.2921 * (10**-5))
# gravitational acceleration
fieldset.add_constant('g', 9.81)
# grid spacing
Delta_x = fieldset.U.grid.lon[1]-fieldset.U.grid.lon[0]
Delta_y = fieldset.U.grid.lat[1]-fieldset.U.grid.lat[0]
Delta_t = dt_timestep

# stepsize for finite differences calculation
delta_x = 0.5 * Delta_x
delta_y = 0.5 * Delta_y
print(f'delta_x = {delta_x}')

fieldset.add_constant('delta_x', delta_x)
fieldset.add_constant('delta_y', delta_y)

lon_min = fieldset.U.grid.lon[1]
lat_min = fieldset.U.grid.lat[1]
lon_max = fieldset.U.grid.lon[-2]
lat_max = fieldset.U.grid.lat[-2]
print(f'lon domain = {lon_min} - {lon_max}')
print(f'lat domain = {lat_min} - {lat_max}')

fieldset.add_constant('lon_min', lon_min)
fieldset.add_constant('lon_max', lon_max)

fieldset.add_constant('lat_min', lat_min)
fieldset.add_constant('lat_max', lat_max)


inertialparticle = InertialParticle2D
land_mask_file = land_directory + 'NWS_mask_land_old.nc'
doggersbank_mask_file = land_directory + 'NWS_mask_doggersbank_old.nc' 
norwegian_trench_mask_file = land_directory + 'NWS_mask_norwegian_trench_old.nc' 
norwegian_trench_hex_file = land_directory + 'NWS_hex_release_norwegian_trench_old.nc' 
NWES_hex_file = land_directory + 'NWES_hex_release_new_2.nc' 
if(starttime >= start_new_dataset):
    land_mask_file = land_directory + 'NWS_mask_land_new.nc'
    doggersbank_mask_file = land_directory + 'NWS_mask_doggersbank_new.nc' 
    norwegian_trench_mask_file = land_directory + 'NWS_mask_norwegian_trench_new.nc' 
    NWES_hex_file = land_directory + 'NWES_hex_release_new_2.nc' 

# doggersbank_mask_file= land_directory +' NWS_mask_doggersbank.nc' # still needs to be created use depth
if (loc == 'custom'):
    lon_particles, lat_particles = set_particles_region(land_mask_file,
                                                        startlon_release,
                                                        endlon_release,
                                                        startlat_release,
                                                        endlat_release)
elif( loc == 'north-sea'):
    lon_particles, lat_particles = set_particles_region(land_mask_file,
                                                        -2,
                                                        9,
                                                        51,
                                                        58)
elif (loc == 'doggersbank'):
    lon_particles, lat_particles = set_particles_region(doggersbank_mask_file,
                                                    1,
                                                    4,
                                                    54,
                                                    56)
elif (loc == 'norwegian-trench'):
    if(grid == 'square'):
        lon_particles, lat_particles = set_particles_region(norwegian_trench_mask_file,
                                                        1,
                                                        10,
                                                        52,
                                                        62)
    elif (grid == 'hex'):
        ds = xr.open_dataset(norwegian_trench_hex_file)
        lon_particles=ds['lon'].values
        lat_particles=ds['lat'].values
    else:
        raise ValueError(f'Error! {grid} should be square or hex')
elif(loc == 'NWES'):
    if(grid == 'square'):
        lon_particles, lat_particles = set_particles_region(land_mask_file,
                                                        fieldset.lon.min(),
                                                        fieldset.lon.max(),
                                                        fieldset.lat.min(),
                                                        fieldset.lat.max())
    elif(grid == 'hex'):
        ds = xr.open_dataset(NWES_hex_file)
        lon_particles=ds['lon'].values
        lat_particles=ds['lat'].values
    else:
        raise ValueError(f'Error! {grid} should be square or hex')
   
else: 
    raise ValueError(f'Error! {loc} should be custom/north-sea/doggersbank/norwegian-trench')

nparticles = lon_particles.size
print(f'nparticles = {nparticles}')
# nparticles = 1  
# lon_particles = 4
# lat_particles = 56
times = np.zeros(nparticles)
Blist = np.full(nparticles, B)
taulist = np.full(nparticles, tau)

# Move particles randomly from starting position 
if (particle_type == 'tracer_random'):
    theta = np.random.rand(nparticles)*2*np.pi
    lon_particles, lat_particles = displace_coordinates(lon_particles,lat_particles,d, theta)



# setting kernels

if (particle_type == 'inertial_initSM'):
    kernels_init = [InitializeParticles2D_MRSM, deleteParticle]
else:
    kernels_init = [InitializeParticles2D, deleteParticle]

kernels = [too_close_to_edge]

if (particle_type == 'tracer'):
    kernels.append(AdvectionRK4)
elif (particle_type == 'tracer_random'):
    kernels.append(AdvectionRK4)
elif (particle_type == 'inertial'):
    kernels.append(MRAdvectionRK4_2D)
elif (particle_type == 'inertial_initSM'):
    kernels.append(MRAdvectionRK4_2D)
elif (particle_type == 'inertial_SM'):
    kernels.append(MRSMAdvectionRK4_2D)

else:
    raise ValueError(f'Error! {particle_type} should' +
                    ' be tracer/tracer_random/inertial/inertial_SM')



kernels.append(remove_at_bounds) 



for release_time in release_times:
    print(release_time)
    times = (release_time-starttime).total_seconds()
    pset = ParticleSet.from_list(fieldset, inertialparticle, lon=lon_particles,
                                    lat=lat_particles, time=times,  B=Blist, tau=taulist)
    endtime_particle  = release_time + runtime
    if (particle_type == 'tracer'):
        output_file = output_file_tracer_b.format(particle_type=particle_type,
                                                loc=loc,
                                                y_s=release_time.year,
                                                m_s=release_time.month,
                                                d_s=release_time.day,
                                                y_e=endtime_particle.year,
                                                m_e=endtime_particle.month,
                                                d_e=endtime_particle.day)
    elif (particle_type == 'tracer_random'):
        output_file = output_file_tracer_random_b.format(particle_type=particle_type,
                                                         d = d,
                                                loc=loc,
                                                y_s=release_time.year,
                                                m_s=release_time.month,
                                                d_s=release_time.day,
                                                y_e=endtime_particle.year,
                                                m_e=endtime_particle.month,
                                                d_e=endtime_particle.day)
    else:
        output_file = output_file_b.format(particle_type=particle_type,
                                        loc=loc,
                                        B=int(B * 1000),
                                        tau=int(tau),
                                        y_s=release_time.year,
                                        m_s=release_time.month,
                                        d_s=release_time.day,
                                        y_e=endtime_particle.year,
                                        m_e=endtime_particle.month,
                                        d_e=endtime_particle.day)

    print(output_file)
    pfile = ParticleFile(output_file, pset, outputdt=dt_write,
                        chunks=(nparticles, 100))

    pfile.add_metadata('dt', str(dt_timestep.total_seconds()))
    pfile.add_metadata('delta_x', str(delta_x))  
    pfile.add_metadata('write_dt', str(dt_write.total_seconds()))
    pfile.add_metadata('runtime', str(runtime.total_seconds()))
    pfile.add_metadata('delta_y', str(delta_y))
    pfile.add_metadata("nparticles", nparticles)
    pfile.add_metadata("particle_type", particle_type)



    pset.execute(kernels_init, runtime=dt_timestep, dt=dt_timestep, verbose_progress=True)
    # I want to reset the time but it does not work
    pset.execute([deleteParticle], runtime=dt_timestep, dt=-dt_timestep, verbose_progress=True)

    # run simulation
    pset.execute(kernels, runtime=runtime, dt=dt_timestep, output_file=pfile)
    

print('Simulation finished!')