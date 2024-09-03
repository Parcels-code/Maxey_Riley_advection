"""
date: August 2024
author: Meike Bos <m.f.bos@uu.nl>
Description: Simulation of intertial particles that move according to the
Maxey-Riley equation. We use the MR-equation without Basset history term and
Faxen corrections. In this script we use RK4 advection in 2D or 3D. We can 
advect the particles both with the full MR-equations, where we keep track
of the velocity of the particle, or with the slow-manifold reduction (SM)
of the MR equation.


We perform test simulations using an analytical flowfield given by and created
in release/analytical_flowfields.py. We load this field from a netcdf file

"""

# import needed packages
import numpy as np
from parcels import FieldSet, ParticleSet, ParticleFile
from parcels import AdvectionRK4, AdvectionRK4_3D
# from parcels import Variable
from datetime import datetime, timedelta
from kernels import InertialParticle2D, InertialParticle3D, deleteParticle
from kernels import InitializeParticles2D, InitializeParticles3D
from kernels import MRAdvectionRK4_2D, MRAdvectionRK4_3D
from kernels import MRSMAdvectionRK4_2D, MRSMAdvectionRK4_3D
from decimal import Decimal

# set input and output directories
base_directory = ('/storage/shared/oceanparcels/' +
                  'output_data/data_Meike/MR_advection/')
input_directory = base_directory + 'fieldsets/'
output_directory = base_directory + 'particle_simulations/'
# output_file = (output_directory + '
# inertia_SM_particle_cylinder_steady_flow_asymetric_3D_RK4_B097_tau_1E-1.zarr')
output_file_b = (output_directory + '{particle_type}_particle_{flow}_{dim}'
                 + '_RK4_B{B:04d}_tau{tau:0.0E}.zarr')

output_file_tracer_b = (output_directory + '{particle_type}_particle_{flow}_{dim}_RK4.zarr')

##################################
#       Simulation settings      #
##################################
# options are tracer, inertial (MR) or inertial_SM (MR slow manifold)
particle_type = 'tracer'
# option 3D or 2D
dim = '2D'
# options are kaufmann (2D), symmetric (3D) and asymmetric (3D) or filename
flow = 'kaufmann'
# integration timestep
dt_timestep = timedelta(seconds=10)
# runtime simulation
runtime = timedelta(hours=3)
# write timestep
dt_write = timedelta(seconds=10)
# Buoyancy (rho_particle/rho_fluid)
B = 0.97
# stokes relaxation time
tau = 1.0 * 10**(-1)

# particles on a grid with size Nx * Ny (2D) or Nx * Ny * Nz (3D)
Nx = 1  # numer of particles in x direction
Ny = 1  # number of particles in y direction
Nz = 10  # number of particles in z direction
xmin = 0.334  # starting position in x direction
xmax = xmin  # end position in x direction
ymin = 0  # start grid in y direction
ymax = 0  # end grid in y direction
zmin = 0.10  # start grid in z direction (only for 3D)
zmax = 0.60  # start grid in z direction (only for 3D)

if (flow == 'kaufmann'):
    input_file = input_directory + 'kaufmann_vortex.nc'
elif (flow == 'symmetric'):
    input_file = input_directory + 'cylinder_flow_symmetric_3D.nc'
elif (flow == 'asymmetric'):
    input_file = input_directory + 'cylinder_flow_asymmetric_3D.nc'
else:
    input_file = input_directory + flow + '.nc'

if (particle_type == 'tracer'):
    output_file = output_file_tracer_b.format(particle_type=particle_type,flow=flow,
                                    dim=dim)
else:
    output_file = output_file_b.format(particle_type=particle_type,flow=flow,
                                        dim=dim, B=int(B * 1000), tau=Decimal(tau))

#########################
#       Set Fields      #
#########################
# variables and dimensions (3D)
variables = {'U': 'U',
             'V': 'V'}

dimensions = {'lat': 'lat',
              'lon': 'lon',
              'time': 'time'}

if (dim == '3D'):
    variables['W'] = 'W'
    dimensions['depth'] = 'depth'

fieldset = FieldSet.from_netcdf(input_file, variables, dimensions,
                                mesh='flat', allow_time_extrapolation='True')
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

fieldset.add_constant('delta_x', delta_x)
fieldset.add_constant('delta_y', delta_y)

if (dim == '3D'):
    Delta_z = fieldset.U.grid.depth[1]-fieldset.U.grid.depth[0]
    delta_z = 0.5 * Delta_z
    fieldset.add_constant('delta_z', delta_z)

###################################
#       Initialize particles      #
###################################
nparticles = Nx * Ny * Nz
x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
z = np.linspace(zmin, zmax, Nz)
depth, lats, lons = np.meshgrid(z, y, x, indexing='ij')
times = np.zeros(nparticles)
Blist = np.full(nparticles, B)
taulist = np.full(nparticles, tau)

kernels_init = []
kernels = []
if (dim == '2D'):
    pset = ParticleSet.from_list(fieldset, InertialParticle2D, lon=lons,
                                 lat=lats, time=times, B=Blist, tau=taulist)
    kernels_init.append(InitializeParticles2D)
    if (particle_type == 'tracer'):
        kernels.append(AdvectionRK4)
    elif (particle_type == 'inertial'):
        kernels.append(MRAdvectionRK4_2D)
    elif (particle_type == 'inertial_SM'):
        kernels.append(MRSMAdvectionRK4_2D)
    else:
        raise ValueError(f'Error! {particle_type} should' +
                         ' be tracer/inertial/inertial_SM')
elif (dim == '3D'):
    pset = ParticleSet.from_list(fieldset, InertialParticle3D, lon=lons,
                                 lat=lats, depth=depth, time=times,
                                 B=Blist, tau=taulist)
    kernels_init.append(InitializeParticles3D)
    if (particle_type == 'tracer'):
        kernels.append(AdvectionRK4_3D)
    elif (particle_type == 'inertial'):
        kernels.append(MRAdvectionRK4_3D)
    elif (particle_type == 'inertial_SM'):
        kernels.append(MRSMAdvectionRK4_3D)
    else:
        raise ValueError(f'Error! {particle_type} should' +
                         ' be tracer/inertial/inertial_SM')
else:
    raise ValueError(f'Error! {dim} should be 2D or 3D!')

kernels_init.append(deleteParticle)
kernels.append(deleteParticle)

######################
#  Run Simulation    #
######################
pfile = ParticleFile(output_file, pset, outputdt=dt_write,
                     chunks=(nparticles, 100))
pfile.add_metadata('dt', str(dt_timestep.total_seconds()))
pfile.add_metadata('delta_x', str(delta_x))  
pfile.add_metadata('write_dt', str(dt_write.total_seconds()))
pfile.add_metadata('runtime', str(runtime.total_seconds()))
pfile.add_metadata('delta_y', str(delta_y))
if (dim == '3D'):
    pfile.add_metadata('delta_z', str(delta_z))
pfile.add_metadata("dimensions", dim)
pfile.add_metadata("nparticles", nparticles)
pfile.add_metadata("particle_type", particle_type)


# initialize fields for the intertial particles
if (particle_type != 'tracer'):
    pset.execute(kernels_init, runtime=1, dt=1, verbose_progress=True)

# run simulation
pset.execute(kernels, runtime=runtime, dt=dt_timestep, output_file=pfile)