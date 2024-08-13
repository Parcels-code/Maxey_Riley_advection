from parcels import JITParticle, Field, FieldSet, Variable
from datetime import datetime, timedelta
import math
import numpy as np
from operator import attrgetter

class PlasticParticle(JITParticle): # at this moment this is not a plastic particle, however as I will work towards plastic particles I already named it accordingly
    heigth = Variable('heigth',dtype=np.float32, to_write=False,initial=0.5) # try different options 
    
    ## Terms needed for MR advection:
    diameter = Variable('diameter',dtype=np.float32, to_write=False,initial=0.1)
    tau = Variable('stokes time',dtype=np.float32, to_write=False,initial=0.5) # not sure whether I want to add this on particle level
    #maybe add last 2 variables only if 3d is true 
    B = Variable('Buoyancy',dtype=np.float32, to_write=False,initial=1) #only for 3d otherwise always 1
    w0 = Variable('stokes terminal velocity',dtype=np.float32, to_write=False,initial=0) #only for 3d otherwise always 1
    
    
    ## write ustokes, vstokes and uwec and vwec 
    # ustokes = Variable('ustokes',dtype=np.float32,to_write=True,initial=0)
    # vstokes = Variable('vstokes',dtype=np.float32,to_write=True,initial=0)
    # uwecpoint = Variable('uwecpoint',dtype=np.float32,to_write=True,initial=0)
    # vwecpoint = Variable('vwecpoint',dtype=np.float32,to_write=True,initial=0)

    #path_length = Variable("path_length",initial=0.,dtype=np.float32, to_write=False)
    #prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False, initial=attrgetter('lon')) # the previous longitude
    #prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False, initial=attrgetter('lat')) # the previous longitude
    #u = Variable('u', dtype=np.float32, initial=0)
    #v = Variable('v', dtype=np.float32, initial=0) 

def MR_AdvectionEE_2D(particle,fieldset,time):
    uf = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    vf = fieldset.V[time, particle.depth, particle.lat, particle.lon]
    

def MR_AdvectionEE_3D(particle,fieldset,time):

def MR_AdvectionRK4_2D(particle,fieldset,time):

def MR_AdvectionRK4_3D(particle,fieldset,time):
