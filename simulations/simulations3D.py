"""
date: August 2024
author: Meike Bos <m.f.bos@uu.nl>
Description: Simulation of intertial particles that move according to the 
Maxey-Riley equation. We use the MR-equation without Bassset
history term and Faxen corrections. In this script we will advect 
particles in 3D 

We perform test simulations using an analytical flowfield given by 
and created in the file release/analytical_flowfields.py. We load this field
from a netcdf file saved in .. 

to do:
- set units to geophysical system
- flake8 check
""" 

# import needed packages
import numpy as np
from parcels import FieldSet, ParticleSet, ParticleFile, Variable, AdvectionRK4_3D # there exists no EE advection in 3D?
from datetime import datetime, timedelta
from kernels import InertialParticle3D, InitializeParticles3D, deleteParticle, MRAdvectionEE_3D, MRAdvectionEC_3D , MRSMAdvectionRK4_3D, MRAdvectionRK4_3D#, InitializeParticles3D, deleteParticle

#set input and output directories
base_directory='/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/'
input_directory=base_directory+'fieldsets/'
output_directory=base_directory+'particle_simulations/'


input_file = input_directory + 'cylinder_flow_symmetric_3D.nc'
output_file = output_directory + 'inertia_particle_cylinder_steady_flow_symmetric_3D_SM_RK4.zarr'

#set integration timestep 
dt_timestep=0.1#0.0001#0.1

#########################
#       Set Fields      #ls
#########################
# variables and dimensions (2D)
variables= {   'U': 'U', #uo, vo
                'V': 'V',
                'W': 'W'}

dimensions = {  'lat': 'lat',
                'lon': 'lon',
                'time': 'time',
                'depth':'depth'}


fieldset = FieldSet.from_netcdf(input_file,variables,dimensions, mesh='flat', allow_time_extrapolation = 'False')
fieldset.add_constant('Omega_earth',7.2921 * (10**-5)) #angular velocity earth in radians/second
Delta_x=fieldset.U.grid.lon[1]-fieldset.U.grid.lon[0]
Delta_y=fieldset.U.grid.lat[1]-fieldset.U.grid.lat[0]
Delta_z=fieldset.U.grid.depth[1]-fieldset.U.grid.depth[0]
Delta_t=fieldset.U.grid.time[1]-fieldset.U.grid.time[0]
print("Delta_x = {:0.2f}".format(Delta_x))
print("Delta_y = {:0.2f}".format(Delta_y))
print("Delta_z = {:0.2f}".format(Delta_z))
print("Delta_t = {:0.2f}".format(Delta_t))

if(Delta_t < dt_timestep):
    raise ValueError('dt_timestep particle larger then temporaral resolution fieldset! Decrease dt_timestep particle')

delta_x=0.5*Delta_x
delta_y=0.5*Delta_y
delta_z=0.5*Delta_z
delta_t=0.5*dt_timestep
fieldset.add_constant('delta_x',delta_x)
fieldset.add_constant('delta_y',delta_y)
fieldset.add_constant('delta_z',delta_z)
fieldset.add_constant('delta_t',delta_t)


###################################
#       Initialize particles      #
###################################

## p

Nx=1
Ny=1
Nz=20
nparticles=Nx*Ny*Nz
xmin=0.334
xmax=xmin
ymin=0
ymax=0
zmin=0.1
zmax=0.6
x=np.linspace(xmin,xmax,Nx)
y=np.linspace(ymin,ymax,Ny)
z=np.linspace(zmin,zmax,Nz)
depth, lats, lons=np.meshgrid(z,y,x, indexing='ij')
times=np.zeros(nparticles)

B=np.full(nparticles,0.97)

tau=np.full(nparticles,0.1)

g=9.81
w0=2*(1-B)/(2+2*B)*g

pset = ParticleSet.from_list(fieldset,InertialParticle3D,lon=lons,lat=lats,depth=depth,time=times,  B=B, tau=tau, w0=w0)#, vf_tm=fieldset.V, up=fieldset.U, vp=fieldset.V)

kernels_init=[InitializeParticles3D,deleteParticle]
kernels=[MRSMAdvectionRK4_3D ,deleteParticle]
kernels_tracer=[AdvectionRK4_3D,deleteParticle]


######################
#  Run Simulation    #
######################

runtime=1000#fieldset.U.grid.time[-2]
dt_write=1#dt_timestep
pfile = ParticleFile(output_file, pset, outputdt=dt_write, chunks=(nparticles,100))

pfile.add_metadata('dt',str(dt_timestep))
pfile.add_metadata('delta_t',str(delta_t))
pfile.add_metadata('delta_x',str(delta_x))
pfile.add_metadata('delta_y',str(delta_y))
pfile.add_metadata('delta_z',str(delta_z))
pfile.add_metadata("dimensions","3D")
pfile.add_metadata("nparticles",nparticles)


# # initialize fields for the particles
dt=0
starttime=0
endtime=0
pset.execute(kernels_init, runtime=1, dt=1,verbose_progress=True)

# run simulation
pset.execute(kernels, runtime=runtime, dt=dt_timestep, output_file=pfile)
