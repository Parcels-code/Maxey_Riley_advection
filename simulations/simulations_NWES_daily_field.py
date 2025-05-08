"""
date: March 2025
author: Meike Bos <m.f.bos@uu.nl>
Description: simulation of "Stokes drifters" in daily averaged fields
(based on file simulations_NWES_repeated_release.py) but now using daily averaged detided fiels
to test effect of time resulution on advection with mr eq
"""

# import needed packages
import click 
import numpy as np
import xarray as xr
from parcels import FieldSet, ParticleSet, ParticleFile, Variable
from parcels import AdvectionRK4
from parcels.tools.converters import Geographic, GeographicPolar 
from datetime import datetime, timedelta
from helper import create_filelist, set_particles_region, displace_coordinates
from kernels import InertialParticle2D, deleteParticle
from kernels import InitializeParticles2D, InitializeParticles2D_MRSM
from kernels import MRAdvectionRK4_2D, MRSMAdvectionRK4_2D
from kernels import MRAdvectionRK4_2D_drag_Rep, MRSMAdvectionRK4_2D_drag_Rep
from kernels import MRSMAdvectionRK4_2D_drag_Rep_constant, MRAdvectionRK4_2D_drag_Rep_constant
from kernels import too_close_to_edge, remove_at_bounds, measure_slip_velocity_SM
from kernels import measure_slip_velocity, measure_slip_velocity_SM
from kernels import set_displacement


@click.command()
@click.option('--rep',default=0, help ='Particle Reynolds number')
@click.option('--year',default=2023,help='starting year')
@click.option('--month',default=9,help='starting month')
@click.option('--day',default=1,help='starting day')
@click.option('--pt',default='tracer',help='particle type')

def run_experiment(pt, rep,year,month,day):
    # set directories
    Rep = rep
    field_directory = '/nethome/4291387/Maxey_Riley_advection/Maxey_Riley_advection/input_data/'
    land_directory = ('/storage/shared/oceanparcels/'
                    'output_data/data_Meike/NWES/')
    output_directory = ('/storage/shared/oceanparcels/'
                        'output_data/data_Meike/MR_advection/NWES/')

    output_file_b = (output_directory + '{particle_type}/{loc}_'
                    'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                    'tres_{time_resolution}_'
                    'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_'
                    'B{B:04d}_tau{tau:04d}_{land_handling}_cor_{coriolis}.zarr')
    
    output_file_Rep_b = (output_directory + '{particle_type}/{loc}_'
                    'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                    'tres_{time_resolution}_'
                    'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_'
                    '_Rep_{Rep:04d}_B{B:04d}_tau{tau:04d}_{land_handling}_cor_{coriolis}.zarr')

    output_file_tracer_random_b = (output_directory + '{particle_type}/{loc}_'
                            'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                            'tres_{time_resolution}_'
                            'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_d{d:04d}_{land_handling}.zarr')
    
    output_file_tracer_b = (output_directory + '{particle_type}/{loc}_'
                            'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                            'tres_{time_resolution}_'
                            'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_{land_handling}.zarr')
    np.random.seed(0)
    ##################################
    #       Simulation settings      #
    ##################################
    # options are tracer, inertial, inertial_SM, inertial_drag_Rep, inertial_SM_drag_Rep
    particle_type = pt #'inertial_drag_Rep'
    land_handling = 'anti_beaching'
    time_resolution = 'daily'
    loc = 'NWES'
    grid = 'hex' 
    save_vorticity = False
    save_fluid_velocity = False
    save_slip_velocity = False
    if( pt in ('inertial_drag_Rep', 'inertial_SM_drag_Rep')):
        save_slip_velocity = True
    coriolis = True
    gradient = True
    starttime = datetime(2023, 9, 1, 0, 0, 0, 0)
    release_time = datetime(year, month, day, 0, 0, 0, 0)
    runtime = timedelta(days=30)  
    endtime = release_time+runtime+timedelta(days=1)
    dt_timestep = timedelta(minutes=5)
    d = 300 
    dt_write =timedelta(hours=1)
    B = 0.68
    tau = 2994.76 # 2759.97
    def factor_drag(Rep):
        c_Rep = 1 + Rep / (4. * (1 +  np.sqrt(Rep))) + Rep / 60.
        return c_Rep
    C_Rep = factor_drag(Rep)
    print(f'C(Rep = {Rep}) = {C_Rep}')

    # Rep = 457#300 # 1000 # 5000 

    #########################
    #       Set Fields      #
    #########################

    # variables = {'U': 'uo_detided',
    #             'V': 'vo_detided'}
    
    variables = {'U': 'uo',
            'V': 'vo'}

    dimensions = {'lat': 'latitude',
                    'lon': 'longitude',
                    'time': 'time'}
    indices = {}
    input_filename = 'cmems_mod_nws_phy_anfc_0.027deg-daily_field_2023-09-01_2024-03-06.nc'#'cmems_mod_nws_phy_anfc_0.027deg-3D_P1D-m_1743500428567.nc' #'cmems_mod_nws_phy-cur_anfc_detided-0.027deg_P1D-m_1741593127167.nc'

    fieldset = FieldSet.from_netcdf(field_directory+input_filename, variables, dimensions, indices=indices,
                                            allow_time_extrapolation="False")

    if (land_handling == 'anti_beaching'):    
        antibeachingfile = land_directory + 'anti_beaching_NWES_daily.nc'
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

    fieldset.add_constant('Rearth', 6371 * 10**3)
    fieldset.add_constant('nu',1.3729308666017527*10**(-6))
    fieldset.add_constant('gradient', gradient)
    fieldset.add_constant('save_slip_velocity',save_slip_velocity)
    # grid spacing
    Delta_x = fieldset.U.grid.lon[1]-fieldset.U.grid.lon[0]
    Delta_y = fieldset.U.grid.lat[1]-fieldset.U.grid.lat[0]
    Delta_t = dt_timestep

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
    NWES_hex_file = land_directory + 'NWES_daily_hex_release_new.nc' 

    if(loc == 'NWES'):
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
        
    nparticles = lon_particles.size
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
    if(save_fluid_velocity == True):
        if(particle_type in ('inertial_SM_Rep_constant', 'inertial_SM_drag_Rep')):
            kernels.append(measure_slip_velocity_SM)
        else:
            kernels.append(measure_slip_velocity)
    # if(save_slip_velocity == True):
    #     kernels.append(measure_slip_velocity)
    if (particle_type == 'tracer'):
        kernels.append(AdvectionRK4)
    elif (particle_type == 'tracer_random'):
        kernels.append(AdvectionRK4)
    elif (particle_type == 'inertial'):
        kernels.append(MRAdvectionRK4_2D)
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
        print('save uslip and vslip')
        setattr(inertialparticle, 'uslip',
            Variable('uslip', dtype=np.float32, to_write=True, initial=0))
        setattr(inertialparticle, 'vslip',
            Variable('vslip', dtype=np.float32, to_write=True, initial=0))
        
    if (save_slip_velocity == True):
        print('save uslip and vslip')
        setattr(inertialparticle, 'uslip',
            Variable('uslip', dtype=np.float32, to_write=True, initial=0))
        setattr(inertialparticle, 'vslip',
            Variable('vslip', dtype=np.float32, to_write=True, initial=0))

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
                                                   time_resolution = time_resolution,
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
                                                         time_resolution = time_resolution,
                                                         save_vorticity = save_vorticity)
    elif(particle_type in ['inertial_SM_Rep_constant','inertial_Rep_constant']):
        output_file = output_file_Rep_b.format(particle_type=particle_type,
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
                                               save_vorticity = save_vorticity,
                                               time_resolution = time_resolution,
                                               Rep = Rep)
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
                                            save_vorticity = save_vorticity,
                                            time_resolution = time_resolution,
                                            Rep = int(Rep))
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
    pset.execute(kernels, runtime=runtime, dt=dt_timestep, output_file=pfile,verbose_progress=False)
        

    print('Simulation finished!')


if __name__ == '__main__':
    run_experiment()
