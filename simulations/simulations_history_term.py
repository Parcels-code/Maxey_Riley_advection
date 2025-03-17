"""
date: march 2025
author: Meike Bos <m.f.bos@uu.nl>
Test simulations for calculation of history term
(release only small fraction of particles with very small timesteps)

"""
# import needed packages
import sys
sys.path.append('../src')

import numpy as np
import xarray as xr
from parcels import FieldSet, ParticleSet, ParticleFile, Variable
from parcels import AdvectionRK4, AdvectionRK4_3D
from parcels.tools.converters import Geographic, GeographicPolar 
import random
random.seed(0) #want to have same sequence of numbers alsways


# from parcels import Variable
from datetime import datetime, timedelta
from helper import create_filelist, set_particles_region, displace_coordinates
from kernels import InertialParticle2D, InertialParticle3D, deleteParticle
from kernels import InitializeParticles2D, InitializeParticles2D_MRSM, InitializeParticles3D
from kernels import MRAdvectionRK4_2D, MRAdvectionRK4_3D, MRAdvectionRK4_2D_Newtonian_drag, MRAdvectionRK4_2D_drag_Rep, MRSMAdvectionRK4_2D_drag_Rep
from kernels import MRSMAdvectionRK4_2D, MRSMAdvectionRK4_3D, MRSMAdvectionRK4_2D_drag_Rep_constant, MRAdvectionRK4_2D_drag_Rep_constant
from kernels import displace, set_displacement, measure_vorticity, measure_fluid_velocity, measure_slip_velocity
from kernels import too_close_to_edge, remove_at_bounds, measure_slip_velocity_SM
from particle_characteristics_functions import factor_drag_white1991

field_directory = ('/storage/shared/oceanparcels/input_data/CopernicusMarineService/'
                   'NORTHWESTSHELF_ANALYSIS_FORECAST_PHY_004_013/')
land_directory = ('/storage/shared/oceanparcels/'
                  'output_data/data_Meike/NWES/')
output_directory = ('/storage/shared/oceanparcels/'
                    'output_data/data_Meike/MR_advection/NWES/test_history_term/')


output_file_b = (output_directory + '{particle_type}/{loc}_'
                 'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                 'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_'
                 'B{B:04d}_tau{tau:04d}_{land_handling}_cor_{coriolis}_dt_{dt_seconds:04d}.zarr')


output_file_tracer_b = (output_directory + '{particle_type}/{loc}_'
                        'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                        'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_{land_handling}_cor_{coriolis}_dt_{dt_seconds:04d}.zarr')


##################################
#       Simulation settings      #
##################################

# options are tracer, tracer_random, inertial (MR) or inertial_SM (MR slow manifold), inertial_initSM (MR velocity initialized using the MR slow manifold eq), # inertial_drag_REp
particle_type = 'inertial_SM'#'inertial_drag_Rep'#'inertial_drag_Rep' 'inertial_SM_drag_Rep' #'inertial_Rep_constant' #'inertial_SM_drag_REp'#'inertial_SM_drag_REp'# 'inertial_drag_REp'
# starting date 
starttime = datetime(2023, 9, 1, 0, 0, 0, 0)
release_times = np.array([  datetime(2023, 9, 1, 0, 0, 0, 0) ])

runtime = timedelta(hours=2) 
endtime = release_times[-1]+runtime+timedelta(days=1) 
dt_timestep = timedelta(seconds=10)
dt_write =timedelta(seconds=10)

B = 0.68
tau = 2759.97
Rep = 457
C_Rep = factor_drag_white1991(Rep)

land_handling = 'anti_beaching'

save_vorticity = False
save_fluid_velocity = True
coriolis = True


loc = 'NWES' #'NWES' #'north-sea'
# options for grid (only does something for norwegian trench at the moment)
# are square for which we use the gridpoint of the velocity field or hexagonal 
grid = 'hex' 


startlon_release = -15#-15
endlon_release = 9 #8# 9
startlat_release = 47# 47
endlat_release = 60#60

indices = {}

#########################
#       Set Fields      #
#########################
start_new_dataset = datetime(2023, 9, 1, 0, 0, 0, 0)
dt_field = timedelta(days=1)
# variables and dimensions (3D)
print('use new dataset (Nologin Spain)')
variables = {'U': 'uo',
             'V': 'vo'}

dimensions = {'lat': 'latitude',
                'lon': 'longitude',
                'time': 'time'}


dt_name_field = timedelta(days=1)
input_filename = ('CMEMS_v6r1_NWS_PHY_NRT_NL_01hav3D_'
                    '{year_t:04d}{month_t:02d}{day_t:02d}_'
                    '{year_t:04d}{month_t:02d}{day_t:02d}_'
                    'R{year_tplus:04d}{month_tplus:02d}{day_tplus:02d}_HC01.nc')

    
oceanfiles=create_filelist(field_directory, input_filename,
                               starttime, endtime, dt_field, dt_name_field)

fieldset = FieldSet.from_netcdf(oceanfiles, variables, dimensions, indices=indices,
                                        allow_time_extrapolation="False")

if(land_handling == 'free_slip'):
    fieldset.interp_method = 'freeslip'
    fieldset.U.interp_method = 'freeslip'
    fieldset.V.interp_method = 'freeslip'

if(land_handling == 'partialslip'):
    fieldset.interp_method = 'partialslip'
    fieldset.U.interp_method = 'partialslip'
    fieldset.V.interp_method = 'partialslip'

if (land_handling == 'anti_beaching'):    
    antibeachingfile = land_directory + 'anti_beaching_NWES_new.nc'
    
    filenames_anti_beaching = {'dispU': antibeachingfile,
                            'dispV': antibeachingfile,
                            'landmask': antibeachingfile,
                            'distance2shore':antibeachingfile}
    dimensions_anti_beaching = {'lat': 'lat',
                                'lon': 'lon'}

    variables_anti_beaching =  {'dispU': 'dispU',
                    'dispV': 'dispV',
                    'landmask':'landmask',
                    'distance2shore' : 'distance2shore'}
    fieldset_anti_beaching = FieldSet.from_netcdf(filenames_anti_beaching,
                                                  variables_anti_beaching,
                                                  dimensions_anti_beaching,
                                                  indices=indices,
                                                  mesh='spherical',
                                                  allow_time_extrapolation="True")
    fieldset_anti_beaching.dispU.units = GeographicPolar()
    fieldset_anti_beaching.dispV.units = Geographic()

    fieldset.add_field(fieldset_anti_beaching.dispU)
    fieldset.add_field(fieldset_anti_beaching.dispV)
    fieldset.add_field(fieldset_anti_beaching.landmask)
    fieldset.add_field(fieldset_anti_beaching.distance2shore)

    # angular velocity earth in radians/second
if(coriolis==True):
    fieldset.add_constant('Omega_earth', 7.2921 * (10**-5))
elif(coriolis==False):
    fieldset.add_constant('Omega_earth', 0)
else:
    ValueError(f"{coriolis} should be True or False")

# gravitational acceleration
fieldset.add_constant('g', 9.81)
#radius earth in meters
fieldset.add_constant('Rearth', 6371 * 10**3)
# kinematic viscosity water
fieldset.add_constant('nu',1.3729308666017527*10**(-6))


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

###################################
#       Initialize particles      #
###################################
inertialparticle = InertialParticle2D
if (land_handling == 'anti_beaching'):
    setattr(inertialparticle, 'dU',
            Variable('dU', dtype=np.float32, to_write=False, initial=0))
    setattr(inertialparticle, 'dV',
            Variable('dV', dtype=np.float32, to_write=False, initial=0))
    setattr(inertialparticle, 'd2s',
            Variable('d2s', dtype=np.float32, to_write=False, initial=1e3))
    
# add particle diameter to field
if (particle_type in ('inertial_drag_Rep','inertial_SM_drag_Rep')):
        setattr(inertialparticle, 'diameter',
            Variable('diameter', dtype=np.float32, to_write=False, initial=0.2))
if (particle_type in ('inertial_Rep_constant', 'inertial_SM_Rep_constant')):
            setattr(inertialparticle, 'diameter',
            Variable('C_Rep', dtype=np.float32, to_write=False, initial=C_Rep))



land_mask_file = land_directory + 'NWS_mask_land_new.nc'
doggersbank_mask_file = land_directory + 'NWS_mask_doggersbank_new.nc' 
norwegian_trench_mask_file = land_directory + 'NWS_mask_norwegian_trench_new.nc' 
NWES_hex_file = land_directory + 'NWES_hex_release_new_2.nc' 





if(loc == 'NWES'):
    if(grid == 'hex'):
        ds = xr.open_dataset(NWES_hex_file)
        lon_particles=ds['lon'].values
        lat_particles=ds['lat'].values
nparticles = lon_particles.size
select = random.sample(range(0, nparticles), 1000) # 100 particles
nparticles = 1000
lon_particles = lon_particles[select]
lat_particles = lat_particles[select]
times = np.zeros(nparticles)
Blist = np.full(nparticles, B)
taulist = np.full(nparticles, tau)

# setting kernels
if (particle_type == 'inertial_initSM'):
    kernels_init = [InitializeParticles2D_MRSM, deleteParticle]
else:
    kernels_init = [InitializeParticles2D, deleteParticle]

kernels = [too_close_to_edge]
if(land_handling == 'anti_beaching'):
    kernels.append(displace)
if(save_vorticity == True):
    kernels.append(measure_vorticity)
if(save_fluid_velocity == True):
    if(particle_type in ('inertial_SM_Rep_constant', 'inertial_SM_drag_Rep')):
        kernels.append(measure_slip_velocity_SM)
    else:
        kernels.append(measure_slip_velocity)
if (particle_type == 'tracer'):
    kernels.append(AdvectionRK4)
elif (particle_type == 'tracer_random'):
    kernels.append(AdvectionRK4)
elif (particle_type == 'inertial'):
    kernels.append(MRAdvectionRK4_2D)
elif (particle_type == 'inertial_Newton'):
    kernels.append(MRAdvectionRK4_2D_Newtonian_drag)
elif (particle_type == 'inertial_drag_Rep'):
    kernels.append(MRAdvectionRK4_2D_drag_Rep)
elif(particle_type == 'inertial_Rep_constant'):
    kernels.append(MRAdvectionRK4_2D_drag_Rep_constant)
elif(particle_type == 'inertial_SM_Rep_constant'):
    kernels.append(MRSMAdvectionRK4_2D_drag_Rep_constant)
elif (particle_type == 'inertial_SM_drag_Rep'):
    kernels.append(MRSMAdvectionRK4_2D_drag_Rep)
elif (particle_type == 'inertial_initSM'):
    kernels.append(MRAdvectionRK4_2D)
elif (particle_type == 'inertial_SM'):
    kernels.append(MRSMAdvectionRK4_2D)

else:
    raise ValueError(f'Error! {particle_type} should' +
                    ' be tracer/tracer_random/inertial/inertial_SM/inertial_SM_v2')


if(land_handling == 'anti_beaching'):
    kernels.append(set_displacement)

kernels.append(remove_at_bounds) 


if (save_fluid_velocity == True):
    print('save uf and vf')
    setattr(inertialparticle, 'uf',
           Variable('uf', dtype=np.float32, to_write=True, initial=0))
    setattr(inertialparticle, 'vf',
           Variable('vf', dtype=np.float32, to_write=True, initial=0))
           

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
                                                d_e=endtime_particle.day,
                                                dt_seconds = dt_write.seconds,
                                                land_handling=land_handling,
                                                coriolis = coriolis, 
                                                save_vorticity = save_vorticity, 
                                                Rep = Rep)
    elif (particle_type == 'tracer_random'):
        output_file = output_file_tracer_random_b.format(particle_type=particle_type,
                                                 d = d,
                                                loc=loc,
                                                y_s=release_time.year,
                                                m_s=release_time.month,
                                                d_s=release_time.day,
                                                y_e=endtime_particle.year,
                                                m_e=endtime_particle.month,
                                                d_e=endtime_particle.day,
                                                land_handling=land_handling, 
                                                coriolis = coriolis,
                                                dt_seconds = dt_write.seconds,
                                                save_vorticity = save_vorticity)
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
                                        d_e=endtime_particle.day,
                                        land_handling=land_handling,
                                        coriolis = coriolis, 
                                        dt_seconds = dt_write.seconds,
                                        save_vorticity = save_vorticity,
                                        Rep = Rep)

    pfile = ParticleFile(output_file, pset, outputdt=dt_write,
                        chunks=(nparticles, 100))

    pfile.add_metadata('dt', str(dt_timestep.total_seconds()))
    pfile.add_metadata('delta_x', str(delta_x))  
    pfile.add_metadata('write_dt', str(dt_write.total_seconds()))
    pfile.add_metadata('runtime', str(runtime.total_seconds()))
    pfile.add_metadata('delta_y', str(delta_y))
    pfile.add_metadata("nparticles", nparticles)
    pfile.add_metadata("particle_type", particle_type)
    pfile.add_metadata('land_handling',land_handling)
    pfile.add_metadata('coriolis',coriolis)


    pset.execute(kernels_init, runtime=1, dt=1, verbose_progress=True)
    # I want to reset the time but it does not work
    pset.execute([deleteParticle], runtime=1, dt=-1, verbose_progress=True)

    # run simulation
    pset.execute(kernels, runtime=runtime, dt=dt_timestep, output_file=pfile)
    
print('Simulation finished!')
