"""
date: July 2025
author: Meike Bos <m.f.bos@uu.nl>
Description: (exploratory) simulations of "stokes drifters"  with Rep>0 in gyre
Simulation of spherical MR-particles with characteristic (size, density) of stokes drifters ((https://metocean.com/products/stokes-drifter/)
we test the effect of advecting with the MR equations for the north atlantic subtropical gyre
this is done because the work of BOM finds that the MR equations cause stronger accomulation floating
debris in the gyre and find that this is an effect which is caused by the MR eqaution. They use system
with zero particle Reynolds number but we think this migth not be realisic
so in this simulation particles are released homogenously and we test
- slow manifold MR with Rep=0
- slow manifold MR with Rep flexible (
- tracer particles
- maybe with and without coriolis 
"""

# import needed packages
import click 
import numpy as np
from parcels import FieldSet, ParticleSet, ParticleFile, Variable
from parcels import AdvectionRK4, AdvectionRK4_3D
from parcels.tools.converters import Geographic, GeographicPolar 
# from parcels import Variable
from datetime import datetime, timedelta
from helper import create_filelist, set_particles_region, displace_coordinates
from kernels import InertialParticle2D, InertialParticle3D, deleteParticle
from kernels import InitializeParticles2D, InitializeParticles2D_MRSM
from kernels import MRSMAdvectionRK4_2D_drag_Rep, MRSMAdvectionRK4_2D_drag_Rep_constant
from kernels import displace, set_displacement, measure_vorticity, measure_fluid_velocity, measure_slip_velocity
from kernels import too_close_to_edge, remove_at_bounds, measure_slip_velocity_SM

@click.command()
@click.option('--rep',default=0, help ='Particle Reynolds number')
@click.option('--year',default=2023,help='starting year')
@click.option('--month',default=9,help='starting month')
@click.option('--day',default=1,help='starting day')
@click.option('--pt',default='tracer',help='particle type')

def run_experiment(pt, rep,year,month,day):
    #settings
    Rep = rep
    particle_type = pt
    dt_timestep = timedelta(minutes=30)
    dt_write = timedelta(hours=24)
    dt_timestep_initial = timedelta(minutes=1)
    runtime_initial = timedelta(minutes=30)
    B = 0.68
    tau = 2994.76
    def factor_drag(Rep):
        c_Rep = 1 + Rep / (4. * (1 +  np.sqrt(Rep))) + Rep / 60.
        return c_Rep
    C_Rep = factor_drag(Rep)
    coriolis = True
    gradient = True # False
    save_slip_velocity = False
    if( pt in ('inertial_SM_drag_Rep', 'inertial_SM_Rep_constant')):
        save_slip_velocity = True
    loc = 'custom'
    grid = 'square'
    startlon_release=-59
    endlon_release=-30
    startlat_release=10
    endlat_release=40
    indices = {'lat': range(1450,3059),
            'lon': range(2200,3400)
            }
    

    land_handling = None # c grid
    starttime = datetime(year,month,day)
    runtime = timedelta(days=365)
    endtime = starttime + runtime
    dtime_data = timedelta(days=1)    #set time resolution data manually (cannot be minutes)
    dtime_execute = timedelta(minutes=10)

    output_directory = ('/storage/shared/oceanparcels/'
                        'output_data/data_Meike/MR_advection/NA_gyre/')
    
    output_file_b = (output_directory + '{particle_type}/{loc}_'
                    'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                    'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_'
                    'B{B:04d}_tau{tau:04d}_cor_{coriolis}_gradient_{gradient}.zarr')

    output_file_Rep_b = (output_directory + '{particle_type}/{loc}_'
                    'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                    'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_'
                    '_Rep_{Rep:04d}_B{B:04d}_tau{tau:04d}_cor_{coriolis}_gradient_{gradient}.zarr')


    output_file_tracer_b = (output_directory + '{particle_type}/{loc}_'
                            'start{y_s:04d}_{m_s:02d}_{d_s:02d}_'
                            'end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4.zarr')


    #Loading the physical dataset and matching the grid
    directory_phy =  '/storage/shared/oceanparcels/input_data/MOi/'
    phy_base_file =  directory_phy + 'GLO12/psy4v3r1-daily_{vector:s}_{y:04d}-{m:02d}-{d:02d}.nc' 
    phy_files_U = []
    phy_files_V = []
    time = starttime
    phy_file_W = directory_phy + 'GLO12/psy4v3r1-daily_{vector:s}_{y:04d}-{m:02d}-{d:02d}.nc'.format(vector = 'W', y = time.year, m = time.month, d = time.day)
    while(time <= endtime):
        phy_files_U.append(phy_base_file.format(vector = 'U', y = time.year, m = time.month, d = time.day))
        phy_files_V.append(phy_base_file.format(vector = 'V', y = time.year, m = time.month, d = time.day))
        time+=dtime_data

    land_mask_file = phy_files_U[0]

    mesh_file_h = directory_phy + "domain_ORCA0083-N006/PSY4V3R1_mesh_hgr.nc"   #Hiermee converteren we het grid!
    mesh_file_z = directory_phy + "domain_ORCA0083-N006/PSY4V3R1_mesh_zgr.nc"
    filenames_phy = {'U': {
            'lon': mesh_file_h,
            'lat': mesh_file_h,
            'depth':phy_file_W,
            'data': phy_files_U,
        },
        'V': {
            'lon': mesh_file_h,
            'lat': mesh_file_h,
            'depth':phy_file_W,
            'data': phy_files_V,
        }
    }
    variables_phy = {'U': 'vozocrtx',
                    'V': 'vomecrty'}

    c_grid_dimensions = {   'lat': 'gphif',
                            'lon': 'glamf',
                            'depth':'depthw',
                            'time': 'time_counter'}

    dimensions_phy = {'U': c_grid_dimensions,
                    'V': c_grid_dimensions}
    
    #Defining a range of indices to load as fieldset to reduce computational time


    #Creating fieldset
    fieldset = FieldSet.from_nemo(filenames=filenames_phy,variables=variables_phy,dimensions=dimensions_phy, indices=indices)
    print('fieldset = made')

    # add constants to fieldset
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
    # turn on or off gradient calculation
    fieldset.add_constant('gradient', gradient)

    #save slip veocty
    fieldset.add_constant('save_slip_velocity',save_slip_velocity)

    # grid spacing
    Delta_x = fieldset.U.grid.lon[0,1]-fieldset.U.grid.lon[0,0]
    Delta_y = fieldset.V.grid.lat[1,0]-fieldset.V.grid.lat[0,0]
    Delta_t = dt_timestep

    delta_x = 0.5 * Delta_x
    delta_y = 0.5 * Delta_y


    fieldset.add_constant('delta_x', delta_x)
    fieldset.add_constant('delta_y', delta_y)
    


    lon_min = fieldset.U.grid.lon[0,1]
    lat_min = fieldset.U.grid.lat[1,0]
    lon_max = fieldset.V.grid.lon[0,-2]
    lat_max = fieldset.V.grid.lat[-2,0]
    print(f'lon domain = {lon_min} - {lon_max}')
    print(f'lat domain = {lat_min} - {lat_max}')

    fieldset.add_constant('lon_min', lon_min)
    fieldset.add_constant('lon_max', lon_max)

    fieldset.add_constant('lat_min', lat_min)
    fieldset.add_constant('lat_max', lat_max)

    inertialparticle = InertialParticle2D
    # add particle diameter to field
    if (particle_type in ('inertial_drag_Rep','inertial_SM_drag_Rep')):
            setattr(inertialparticle, 'diameter',
                Variable('diameter', dtype=np.float32, to_write=False, initial=0.2))
    if (particle_type in ('inertial_Rep_constant', 'inertial_SM_Rep_constant')):
                setattr(inertialparticle, 'diameter',
                Variable('C_Rep', dtype=np.float32, to_write=False, initial=C_Rep))

    release_spacing = 1 # degrees
    lons = np.arange(startlon_release,endlon_release,release_spacing)
    lats = np.arange(startlat_release,endlat_release,release_spacing)
    lon_particles, lat_particles = np.meshgrid(lons,lats)
    lon_particles = lon_particles.flatten()#[-10:-1]
    lat_particles = lat_particles.flatten()#[-10:-1]
    # print(lon_particles)
    # print(lat_particles)

    nparticles = lon_particles.size 
    times = np.zeros(nparticles)
    Blist = np.full(nparticles, B)
    print(f'nparticles = {nparticles}')
    taulist = np.full(nparticles, tau)


    #setting kernels 
    kernels_init = [InitializeParticles2D, deleteParticle]
    kernels = [too_close_to_edge]
    if (particle_type == 'tracer'):
        kernels.append(AdvectionRK4)
    elif(particle_type == 'inertial_SM_Rep_constant'):
        kernels.append(MRSMAdvectionRK4_2D_drag_Rep_constant)
    elif (particle_type == 'inertial_SM_drag_Rep'):
         kernels.append(MRSMAdvectionRK4_2D_drag_Rep)
    else:
         raise ValueError(f'Error! {particle_type} should be tracer/inertial_SM_drag_Rep/inertial_SM_Rep_constant')
    
    setattr(inertialparticle, 'uslip',
            Variable('uslip', dtype=np.float32, to_write=save_slip_velocity, initial=0))
    setattr(inertialparticle, 'vslip',
            Variable('vslip', dtype=np.float32, to_write=save_slip_velocity, initial=0))
    
    pset = ParticleSet.from_list(fieldset, inertialparticle, lon=lon_particles,
                                        lat=lat_particles, time=times,  B=Blist, tau=taulist)
    
    kernels.append(remove_at_bounds) 
    kernels_init= kernels_init + kernels

    
    if (particle_type == 'tracer'):
            output_file = output_file_tracer_b.format(particle_type=particle_type,
                                                    loc=loc,
                                                    y_s=starttime.year,
                                                    m_s=starttime.month,
                                                    d_s=starttime.day,
                                                    y_e=endtime.year,
                                                    m_e=endtime.month,
                                                    d_e=endtime.day,
                                                    coriolis = coriolis, 
                                                    Rep = Rep)
            
    elif(particle_type in ['inertial_SM_Rep_constant','inertial_Rep_constant']):
            output_file = output_file_Rep_b.format(particle_type=particle_type,
                                            loc=loc,
                                            B=int(B * 1000),
                                            tau=int(tau),
                                            y_s=starttime.year,
                                            m_s=starttime.month,
                                            d_s=starttime.day,
                                            y_e=endtime.year,
                                            m_e=endtime.month,
                                            d_e=endtime.day,
                                            coriolis = coriolis, 
                                            Rep = Rep,
                                            gradient = gradient)
    else:
            output_file = output_file_b.format(particle_type=particle_type,
                                    loc=loc,
                                    B=int(B * 1000),
                                    tau=int(tau),
                                    y_s=starttime.year,
                                    m_s=starttime.month,
                                    d_s=starttime.day,
                                    y_e=endtime.year,
                                    m_e=endtime.month,
                                    d_e=endtime.day,
                                    coriolis = coriolis, 
                                    Rep = int(Rep),
                                    gradient = gradient )

    pfile = ParticleFile(output_file, pset, outputdt=dt_write,
                            chunks=(nparticles, 100)) #nparticles
    
    pfile.add_metadata('dt', str(dt_timestep.total_seconds()))
    pfile.add_metadata('delta_x', str(delta_x))  
    pfile.add_metadata('write_dt', str(dt_write.total_seconds()))
    pfile.add_metadata('runtime', str(runtime.total_seconds()))
    pfile.add_metadata('delta_y', str(delta_y))
    pfile.add_metadata("nparticles", nparticles)
    pfile.add_metadata("particle_type", particle_type)
    pfile.add_metadata('land_handling',land_handling)
    pfile.add_metadata('coriolis',coriolis)

    pset.execute(kernels_init, runtime=runtime_initial, dt=dt_timestep_initial, verbose_progress=True,output_file=pfile)


        # run simulation
    pset.execute(kernels, runtime=runtime, dt=dt_timestep,verbose_progress=False,  output_file=pfile)
    print('Simulation finished!')


if __name__ == '__main__':
    run_experiment()

