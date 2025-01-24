"""
date: August 2024
author: Meike Bos <m.f.bos@uu.nl>
Description: Simulation of "stokes drifters" 
(https://metocean.com/products/stokes-drifter/)" that either move as tracer
particles or as inertial particles according to the Maxey-Riley equation.
We release the particles in the North West European shelf (NWES) (CMEMS data).
For the inertial advection we use the slow manifold reduced MR-equations
without Basset history term and Faxen corrections. We use a 2D simulation.

We perform test simulations using an analytical flowfield given by and created
in release/analytical_flowfields.py. We load this field from a netcdf file

The drifters have an outer diamter of 24 cm which we use as radius of the particle. 
The 
"""
# import needed packages
import numpy as np
import xarray as xr
from parcels import FieldSet, ParticleSet, ParticleFile, Variable
from parcels import AdvectionRK4, AdvectionRK4_3D
from parcels.tools.converters import Geographic, GeographicPolar
# from parcels import Variable
from datetime import datetime, timedelta
from helper import create_filelist, set_particles_region, displace_coordinates
from kernels import InertialParticle2D, InertialParticle3D, deleteParticle
from kernels import InitializeParticles2D, InitializeParticles2D_MRSM, InitializeParticles3D
from kernels import MRAdvectionRK4_2D, MRAdvectionRK4_3D, MRAdvectionRK4_2D_Newtonian_drag
from kernels import MRSMAdvectionRK4_2D, MRSMAdvectionRK4_3D
from kernels import displace, set_displacement, measure_vorticity, measure_fluid_velocity, measure_slip_velocity
from kernels import too_close_to_edge, remove_at_bounds

# set directories
field_directory = ('/storage/shared/oceanparcels/input_data/CopernicusMarineService/'
                   'NORTHWESTSHELF_ANALYSIS_FORECAST_PHY_004_013/')
land_directory = ('/storage/shared/oceanparcels/'
                  'output_data/data_Meike/NWES/')
output_directory = ('/storage/shared/oceanparcels/'
                    'output_data/data_Meike/MR_advection/NWES/')
output_file_b = (output_directory + '{particle_type}/{loc}_'
                 'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                 'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_'
                 'B{B:04d}_tau{tau:04d}_{land_handling}_cor_{coriolis}_vorticity_{save_vorticity}.zarr')
output_file_tracer_b = (output_directory + '{particle_type}/{loc}_'
                        'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                        'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_{land_handling}_vorticity_{save_vorticity}.zarr')
output_file_tracer_random_b = (output_directory + '{particle_type}/{loc}_'
                        'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                        'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_d{d:04d}_{land_handling}.zarr')


##################################
#       Simulation settings      #
##################################

# options are tracer, tracer_random, inertial (MR) or inertial_SM (MR slow manifold), inertial_initSM (MR velocity initialized using the MR slow manifold eq)
particle_type = 'inertial_Newton'
# starting date 
starttime = datetime(2023, 9, 1, 0, 0, 0, 0)
release_times = np.array([ datetime(2023, 9, 1, 0, 0, 0, 0)])
#                             datetime(2023, 10, 1, 0, 0, 0, 0),
#                             datetime(2023, 11, 1, 0, 0, 0, 0),
#                             datetime(2023, 12, 1, 0, 0, 0, 0)])
                          #datetime(2024, 1, 1, 0, 0, 0, 0),
                          #datetime(2024, 2, 1, 0, 0, 0, 0)]) 
                    # [datetime(2023, 9, 1, 0, 0, 0, 0),
                    #   datetime(2023, 10, 1, 0, 0, 0, 0)

                    #   datetime(2024, 3, 1, 0, 0, 0, 0),
                    #   datetime(2024, 4, 1, 0, 0, 0, 0)])
# settings for temporal releaste

runtime = timedelta(days=30)
# total_runtime = timedelta(days=10)
# endtime = datetime(2024, 5, 1, 0, 0, 0, 0)#starttime +timedelta(days=45)
endtime = release_times[-1]+runtime+timedelta(days=1)#datetime(2024, 5, 1, 0, 0, 0, 0)
# integration timestep
dt_timestep = timedelta(minutes=5)
# write timestep
dt_write = timedelta(hours=1)
# Buoyancy (rho_particle/rho_fluid)
B = 0.68
# stokes relaxation time
tau = 2759.97
# newton drag length scale
ld = 0.499
#random displacement distance starting position in meters
d = 100 # m 
# 
# set land boundray handling (options: anti_beaching (anti-beaching kernel) or free_slip (free slip fieldset)) or none
land_handling = 'anti_beaching' # 'free_slip' #'anti_beaching' # 'anti_beaching' #partialslip

save_vorticity = False
save_fluid_velocity = True
coriolis = True

# particle release location
# option location is North sea, custom, doggersbank, Norwegian trench
loc = 'custom' #'NWES' #'north-sea'
# options for grid (only does something for norwegian trench at the moment)
# are square for which we use the gridpoint of the velocity field or hexagonal 
grid = 'hex' 
# set custom region:
startlon_release = -15
endlon_release = 9
startlat_release = 47
endlat_release = 60
# Entire North Sea
# startlon_release=1
# endlon_release=7
# startlat_release=51
# endlat_release=58
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
    antibeachingfile = land_directory + 'anti_beaching_NWES_old.nc'
    if(starttime >= start_new_dataset):
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

# fieldset.add_constant('Omega_earth', 0)

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
# nparticles = 1  
# lon_particles = 4
# lat_particles = 56
times = np.zeros(nparticles)
Blist = np.full(nparticles, B)
if(particle_type=='inertial_Newton'):
    taulist = np.full(nparticles,ld)
else:
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
if(land_handling == 'anti_beaching'):
    kernels.append(displace)
if(save_vorticity == True):
    kernels.append(measure_vorticity)
if(save_fluid_velocity == True):
    kernels.append(measure_slip_velocity)
if (particle_type == 'tracer'):
    kernels.append(AdvectionRK4)
elif (particle_type == 'tracer_random'):
    kernels.append(AdvectionRK4)
elif (particle_type == 'inertial'):
    kernels.append(MRAdvectionRK4_2D)
elif (particle_type == 'inertial_Newton'):
    kernels.append(MRAdvectionRK4_2D_Newtonian_drag)
elif (particle_type == 'inertial_initSM'):
    kernels.append(MRAdvectionRK4_2D)
elif (particle_type == 'inertial_SM'):
    kernels.append(MRSMAdvectionRK4_2D)
elif (particle_type == 'inertial_SM_v2'):
    kernels.append(MRSMAdvectionRK4_2D_v2)

else:
    raise ValueError(f'Error! {particle_type} should' +
                    ' be tracer/tracer_random/inertial/inertial_SM/inertial_SM_v2')

if(land_handling == 'anti_beaching'):
    kernels.append(set_displacement)

kernels.append(remove_at_bounds) 

if (save_vorticity == True):
    setattr(inertialparticle, 'vorticity',
           Variable('vorticity', dtype=np.float32, to_write=True, initial=0))

if (save_fluid_velocity == True):
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
                                                land_handling=land_handling,
                                                coriolis = coriolis, 
                                                save_vorticity = save_vorticity)
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
                                        save_vorticity = save_vorticity)

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
