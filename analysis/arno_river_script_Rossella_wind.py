############# IMPORT NEEDED PACKAGES ####################
import numpy as np
import parcels as parcels
from datetime import datetime, timedelta
import xarray as xr
import math
#import matplotlib.pyplot as plt

############################ SIMULATION SETTINGS ############################
# set time at which you want your simulation to start (if your netcdf has
# timestamps with it)

#starttime = datetime(2021, 2, 8, 0, 0, 0)

# make sure that time_res_netcdf/dt_timestep = integer number
dt_timestep = timedelta(seconds=30)

# timestep which with output saved, make sure that this is integer multiple of dt_timestep
dt_output = timedelta(minutes=1)
runtime = timedelta(minutes=1917)  # total runtime of the simulation
nparticles = 10

# repeat particle release every repeaddt time (again make sure that this is
# integer multiple of dt_timestep)
repeatdt = timedelta(minutes=1)
nrelease = int(runtime/repeatdt)

# array with x-coordinates (lons) of start positions particles
lons_particles = np.linspace(6360, 6365, nparticles)

# array with y-coordinates (lats) of start positions particles
lats_particles = np.linspace(3055, 3060, nparticles)

############################# SET DIRECTORIES #############################
# path to directory where the netcdf files are stored
directory_input = "/home/mocali/funwave/output_sanrossore/89feb21/"
# path to directory where you want to store the simulation output
directory_output = "/home/mocali/parcels/san_rossore/89feb21/"

input_filename = directory_input + "waves_89feb21_mean.nc"
output_filename = directory_output + "output_wind.zarr"

############## CREATE FIELDSETS #########################
# velocities of the main fieldset are always called U and V (and W) in parcels
variables = {
    "U": "umean",
    "V": "vmean"}  

# dimensions are always called lat, lon and time in parcels (even if it a flat mesh)
dimensions = {"lat": "y",
              "lon": "x",
              "time": "time"}
			  
# load fieldset
fieldset = parcels.FieldSet.from_netcdf(
    input_filename,
    variables,
    dimensions,
    mesh="flat",
    allow_time_extrapolation=False
)

# wind fieldset
filenames_wind = directory_output + "wind_field_feb8_9_2021.nc"
variables_wind = {'wind_U': 'Uwind', 'wind_V': 'Vwind'}
dimensions_wind = {'time': 'time', 'lat': 'y', 'lon': 'x'}

fieldset_wind = parcels.FieldSet.from_netcdf(filenames_wind,
                                              variables_wind,
                                              dimensions_wind,
                                              mesh='flat')

fieldset.add_field(fieldset_wind.wind_U)
fieldset.add_field(fieldset_wind.wind_V)

#fieldset.wind_U.allow_extrapolation = True
#fieldset.wind_V.allow_extrapolation = True


######## ADD BOUNDARIES DOMAIN TO FIELDSET #############
fieldset.add_constant('lon_min', fieldset.U.grid.lon[0])
fieldset.add_constant('lon_max', fieldset.U.grid.lon[-1])

fieldset.add_constant('lat_min', fieldset.U.grid.lat[0])
fieldset.add_constant('lat_max', fieldset.U.grid.lat[-1])
 
############# CREATE PARTICLES ##########################

PlasticParticle = parcels.JITParticle.add_variable([
    parcels.Variable('beached', dtype=np.int32, initial=0, to_write=False),
    parcels.Variable('wind_coefficient', dtype=np.float32, initial=0.03),      # default 3%
])

pset = parcels.ParticleSet(
    fieldset=fieldset,
    pclass=PlasticParticle,
    lon=lons_particles,
    lat=lats_particles,
    repeatdt=repeatdt
)

# Fieldset Plot
#fieldset.computeTimeChunk()
#plt.pcolormesh(fieldset.U.grid.lon, fieldset.U.grid.lat, fieldset.U.data[0, :, :])
#plt.xlabel("X [m]")
#plt.ylabel("Y [m]")
#plt.colorbar()
#plt.show()

# Plot Particles Release Position
#fieldset.computeTimeChunk()
#plt.pcolormesh(fieldset.U.grid.lon, fieldset.U.grid.lat, fieldset.U.data[0,:,:])
#plt.scatter(pset.lon, pset.lat, color='red')
#plt.xlabel("X [m]")
#plt.ylabel("Y [m]")
#plt.show()


pfile = pset.ParticleFile(
    name=output_filename, outputdt=dt_output, chunks=(nparticles*nrelease, 10)
)

########### SET KERNELS #################################

### custom kernel definitions
def DeleteParticle(particle, fieldset, time):
    """
    Delete particle if they are out of simulation domain
    """
    if particle.state == parcels.StatusCode.ErrorOutOfBounds:
        particle.delete()

# --- kernel WindageDrift ---
def WindageDrift(particle, fieldset, time):
    """
    Apply windage drift as a linear addition to the velocity field
    u(x,t) = u_c(x,t) + C_w * (u_w(x,t)-u_c(x,t))`
    """
    # Sample sea velocities
    (sea_U, sea_V) = fieldset.UV[particle]
    sea_speed = math.sqrt(sea_U**2 + sea_V**2)
    
    # Sample the U / V components of wind
    Wind_U = fieldset.wind_U[time, particle.depth, particle.lat, particle.lon]
    Wind_V = fieldset.wind_V[time, particle.depth, particle.lat, particle.lon]

    # Compute particle displacement
    particle_dlon += particle.wind_coefficient * (Wind_U - sea_U) * particle.dt
    particle_dlat += particle.wind_coefficient * (Wind_V - sea_V) * particle.dt 

def remove_at_bounds(particle, fieldset, time):
    """
    Kernel for deleting particles if they outside the boundary of the
    simulation domaim. 
    
    Dependencies:
    - lon_min,lon_max, lat_min, lat_max, boundaries of domain 
    """
    flag_ = False
    if particle.lat < fieldset.lat_min:
        particle.delete()
        flag_ = True
    if particle.lat > fieldset.lat_max:
        particle.delete()        
        flag_ = True
    if particle.lon < fieldset.lon_min:
        particle.delete()
        flag_ = True
    if particle.lon > fieldset.lon_max:
        particle.delete()
        flag_ = True
     

# AdvectionRK4 is most standard advection in 2d (lon,lat)
kernels = [parcels.AdvectionRK4, DeleteParticle, WindageDrift]

########### RUN SIMULATIONS #############################
# use execute to run the simulation
pset.execute(kernels, runtime=runtime, dt=dt_timestep, output_file=pfile)