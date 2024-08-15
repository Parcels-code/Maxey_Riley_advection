"""
date: August 2024
author: Meike Bos <m.f.bos@uu.nl>
Description: Simulation of intertial particles that move according to the 
Maxey-Riley equation. We use the MR-equation without Bassset
history term and Faxen corrections. We simulate neutrally buoyant particles
in 2D. 

We perform test simulations using an analytical flowfield given by 
and created in the file release/analytical_flowfields.py. WE load this field
from a netcdf file saved in .. 

to do:
- create analytical field
- set units to geophysical system
- flake8 check

"""

# import needed packages
import numpy as np
from parcels import FieldSet, ParticleSet, ParticleFile, Variable
from datetime import datetime, timedelta
from kernels import InertialParticle, MRAdvectionEC2D, InitializeParticles, deleteParticle

from parcels.tools.converters import Geographic, GeographicPolar


#set input and output directories
base_directory='/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/'
input_directory=base_directory+'fieldsets/'
output_directory=base_directory+'particle_simulations/'
# set files 
input_file = input_directory + 'kaufmann_vortex_field.nc'
output_file = output_directory + 'Inertial_particles_MREC2D_Kaufmann_vortex.zarr'

#set integration timestep 
dt_timestep=1

#########################
#       Set Fields      #
#########################
# variables and dimensions (2D)
variables= {   'U': 'U', #uo, vo
                        'V': 'V'}




dimensions = {  'lat': 'lat',
                'lon': 'lon',
                'time': 'time'}

fieldset = FieldSet.from_netcdf(input_file,variables,dimensions, mesh='flat', allow_time_extrapolation = 'False')
fieldset.add_constant('Omega_earth',7.2921 * (10**-5)) #angular velocity earth in radians/second
Delta_x=fieldset.U.grid.lon[1]-fieldset.U.grid.lon[0]
Delta_y=fieldset.U.grid.lat[1]-fieldset.U.grid.lat[0]
Delta_t=fieldset.U.grid.time[1]-fieldset.U.grid.time[0]
print("Delta_x = {:0.2f}".format(Delta_x))
print("Delta_y = {:0.2f}".format(Delta_y))
print("Delta_t = {:0.2f}".format(Delta_t))

if(Delta_t < dt_timestep):
    raise ValueError('dt_timestep particle larger then temporaral resolution fieldset! Decrease dt_timestep particle')
 
delta_x=0.5*Delta_x
delta_t=0.5*dt_timestep
fieldset.add_constant('delta_x',delta_x)
fieldset.add_constant('delta_y',delta_x)
fieldset.add_constant('delta_t',delta_t)


###################################
#       Initialize particles      #
###################################
Nx=10
Ny=10
nparticles=Nx*Ny
xmin=-10
xmax=10
ymin=-10
ymax=10
x=np.linspace(xmin,xmax,Nx)
y=np.linspace(ymin,ymax,Ny)
lons, lats=np.meshgrid(x,y)
times=np.zeros(nparticles)

B=np.full(nparticles,0.9)
Bterm=(3./(1+2*B))
tau=np.full(nparticles,1.*10**-3)
tau_inv=1.0/tau

pset = ParticleSet.from_list(fieldset,InertialParticle,lon=lons,lat=lats,time=times,
                             Bterm=Bterm, tau_inv=tau_inv)#, vf_tm=fieldset.V, up=fieldset.U, vp=fieldset.V)


kernels=[MRAdvectionEC2D,deleteParticle]
kernels_init=[InitializeParticles,deleteParticle]

######################
#  Run Simulation    #
######################

runtime=100#fieldset.U.grid.time[-2]
dt_write=dt_timestep
pfile = ParticleFile(output_file, pset, outputdt=dt_write, chunks=(nparticles,100))

# add simulation settings as metadata
pfile.add_metadata('dt',str(dt_timestep))
pfile.add_metadata('delta_t',str(delta_t))
pfile.add_metadata('delta_x',str(delta_x))
pfile.add_metadata("dimensions","2D")
pfile.add_metadata("nparticles",nparticles)


# initialize fields for the particles
dt=0
starttime=0
endtime=0
pset.execute(kernels_init, runtime=1, dt=1,verbose_progress=True)


# run simulation

pset.execute(kernels, runtime=runtime, dt=dt_timestep, output_file=pfile)



