"""
date: August 2024
author: Meike Bos <m.f.bos@uu.nl>
Description: Simulation of intertial particles that move according to the 
Maxey-Riley equation. We use the MR-equation without Bassset
history term and Faxen corrections. In this script we will advect 
particles in 3D 

We perform test simulations using an analytical flowfield given by 
and created in the file release/analytical_flowfields.py. We load this field
from a netcdf file 

to do:
- flake8 check
""" 

# import needed packages
import numpy as np
from parcels import FieldSet, ParticleSet, ParticleFile, Variable, AdvectionRK4_3D 
from datetime import datetime, timedelta
from kernels import InertialParticle3D, InitializeParticles3D, deleteParticle, MRAdvectionEE_3D, MRAdvectionEC_3D , MRSMAdvectionRK4_3D, MRAdvectionRK4_3D, MRSMAdvectionRK4_2D, MRAdvectionRK4_2D#, InitializeParticles3D, deleteParticle

# set input and output directories
base_directory = '/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/'
input_directory=base_directory+'fieldsets/'
output_directory=base_directory+'particle_simulations/'


input_file = input_directory + 'cylinder_flow_asymmetric_3D.nc'
output_file = output_directory + 'inertia_SM_particle_cylinder_steady_flow_asymetric_3D_RK4_B097_tau_1E-1.zarr'

#timestepping
dt_timestep=timedelta(seconds=1)


#########################
#       Set Fields      #
#########################
# variables and dimensions (2D)
variables= {'U': 'U',
            'V': 'V',
            'W': 'W'}

dimensions = {  'lat': 'lat',
                'lon': 'lon',
                'time': 'time',
                'depth':'depth'}


g=10 

fieldset = FieldSet.from_netcdf(input_file,variables,dimensions, mesh='flat', allow_time_extrapolation = 'True')
fieldset.add_constant('Omega_earth',7.2921 * (10**-5)) #angular velocity earth in radians/second
fieldset.add_constant('g',g) #gravitational acceleration
Delta_x=fieldset.U.grid.lon[1]-fieldset.U.grid.lon[0]
Delta_y=fieldset.U.grid.lat[1]-fieldset.U.grid.lat[0]
Delta_z=fieldset.U.grid.depth[1]-fieldset.U.grid.depth[0]
Delta_t=dt_timestep #fieldset.U.grid.time[1]-fieldset.U.grid.time[0]
print("Delta_x = {:0.3f}".format(Delta_x))
print("Delta_y = {:0.3f}".format(Delta_y))
print("Delta_z = {:0.3f}".format(Delta_z))


# if(Delta_t < dt_timestep):
#     raise ValueError('dt_timestep particle larger then temporaral resolution fieldset! Decrease dt_timestep particle')

delta_x = 0.5*Delta_x
delta_y = 0.5*Delta_y
delta_z = 0.5*Delta_z
delta_t = 0.5*dt_timestep
fieldset.add_constant('delta_x',delta_x)
fieldset.add_constant('delta_y',delta_y)
fieldset.add_constant('delta_z',delta_z)
fieldset.add_constant('delta_t',delta_t)


###################################
#       Initialize particles      #
###################################
Nx = 1
Ny = 1
Nz = 10
nparticles = Nx * Ny * Nz
xmin= 0.334
xmax= xmin
ymin = 0
ymax = 0
zmin = 0.10
zmax = 0.60
x = np.linspace(xmin,xmax,Nx)
y = np.linspace(ymin,ymax,Ny)
z = np.linspace(zmin,zmax,Nz)
depth, lats, lons = np.meshgrid(z,y,x, indexing='ij')
times=np.zeros(nparticles)

B = np.full(nparticles,0.97)

tau = np.full(nparticles,0.1)
pset = ParticleSet.from_list(fieldset,InertialParticle3D,lon=lons,lat=lats,depth=depth,time=times,  B=B, tau=tau) 

kernels_init=[InitializeParticles3D,deleteParticle]
kernels = [MRSMAdvectionRK4_3D ,deleteParticle]
kernels_tracer = [AdvectionRK4_3D,deleteParticle]

######################
#  Run Simulation    #
######################

runtime = timedelta(hours=24) 
dt_write = timedelta(minutes=5) 
pfile = ParticleFile(output_file, pset, outputdt=dt_write, chunks=(nparticles,100))
pfile.add_metadata('dt',str(dt_timestep.total_seconds()))
pfile.add_metadata('delta_t',str(delta_t))
pfile.add_metadata('delta_x',str(delta_x))
pfile.add_metadata('write_dt',str(dt_write.total_seconds()))
pfile.add_metadata('runtime', str(runtime.total_seconds()))
pfile.add_metadata('delta_y',str(delta_y))
pfile.add_metadata('delta_z',str(delta_z))
pfile.add_metadata("dimensions","3D")
pfile.add_metadata("nparticles",nparticles)


# initialize fields for the particles
dt=0
starttime=0
endtime=0
pset.execute(kernels_init, runtime=1, dt=1,verbose_progress=True)

# run simulation
pset.execute(kernels, runtime=runtime, dt=dt_timestep, output_file=pfile)
