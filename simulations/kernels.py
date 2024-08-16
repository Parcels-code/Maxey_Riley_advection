from parcels import JITParticle, Field, FieldSet, Variable
from datetime import datetime, timedelta
import math
import numpy as np
from operator import attrgetter

class InertialParticle(JITParticle): # particle that is avected according to the MR equation
    ## Terms needed for MR advection:
    tau_inv = Variable('tau_inv',dtype=np.float32, to_write='once',initial=1000) 
    uf_tm = Variable('uf_tm',dtype=np.float32, to_write=False,initial=0) #needed because parcels loads in only 2 timesteps of the velocity field
    vf_tm = Variable('vf_tm',dtype=np.float32, to_write=False,initial=0) #needed because parcels loads in only 2 timesteps of the velocity field
    up= Variable('up',dtype=np.float32, to_write=False, initial=0) #better to sample with initial velocity field?
    vp= Variable('vp',dtype=np.float32, to_write=False,initial=0)
    Bterm  = Variable('Bterm',dtype=np.float32, to_write='once',initial=1)

    # only add following variables if 3D MR advection (will be implemented later)
    #B = Variable('Buoyancy',dtype=np.float32, to_write=False,initial=1) #only for 3d otherwise always 1
    # w0 = Variable('stokes terminal velocity',dtype=np.float32, to_write=False,initial=0) #only for 3d otherwise always 1
    # wf_tm = Variable('vtm',dtype=np.float32, to_write=False,initial=0)
    # wp = Variable()
    
    ## for testing write gradients and veloctiy
    #dudt = Variable('dudt',dtype=np.float32, to_write=True, initial=0)
    #DuDt = Variable('DuDt',dtype=np.float32, to_write=True, initial=0)
    
def InitializeParticles(particle,fieldset,time): 
    (u,v)=fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    particle.up=u
    particle.vp=v
    particle.uf_tm=u
    particle.vf_tm=v

def MRAdvectionEC2D(particle,fieldset,time): 
    """
    Advection of particles using Maxey-Riley equation in 2D
    without Basset history term and Faxen corrections 
    without sinking or floating force. the equation is numerically
    integrated using the Euler-Cromes scheme

    dependencies:
    the particle should contain variables uf_tm, vf_tm(u and v 
    component velocity sampled at previous timestep), 
    up, vp (velocity particle at timestep t),tau_inv is 
    1/stokes time and Bterm=3/(1+2B) with B the buoyancy
    of theparticle. 
    The fieldset should contain constants Omega_earth (angular 
    velocity earth), delta_x (stepping for gradients), delta_t 
    stepping for temporal derivatives)

    TO DO: double check formula (and prefactors)

    """
    # read in velocity at location of particle
    uf,vf= fieldset.UV[time, particle.depth, particle.lat, particle.lon]

    # calculate time derivative of fluid field
    (uf_tp, vf_tp) = fieldset.UV[time+fieldset.delta_t, particle.depth, particle.lat, particle.lon]
    norm_deltat=1./(2.0*fieldset.delta_t)
    dudt=(uf_tp-particle.uf_tm)*norm_deltat
    dvdt=(vf_tp-particle.vf_tm)*norm_deltat
    
    # calculate spatial gradients fluid field
    (u_dxm,v_dxm)=fieldset.UV[time, particle.depth, particle.lat, particle.lon-fieldset.delta_x]
    (u_dxp,v_dxp)=fieldset.UV[time, particle.depth, particle.lat, particle.lon+fieldset.delta_x]
    (u_dym,v_dym)=fieldset.UV[time, particle.depth, particle.lat-fieldset.delta_y, particle.lon]
    (u_dyp,v_dyp)=fieldset.UV[time, particle.depth, particle.lat+fieldset.delta_y, particle.lon]
    norm_deltax=1.0/(2.0*fieldset.delta_x)
    norm_deltay=1.0/(2.0*fieldset.delta_y) 
    dudx=(u_dxp-u_dxm)*norm_deltax
    dudy=(u_dyp-u_dym)*norm_deltay
    dvdx=(v_dxp-v_dxm)*norm_deltax
    dvdy=(v_dyp-v_dym)*norm_deltay

    # use time derivative and spatial gradients to caluclate material derivative fluid
    DuDt=dudt+uf*dudx+vf*dudy
    DvDt=dvdt+uf*dvdx+vf*dvdy

    # coriolis force
    f=0#2*fieldset.Omega_earth*math.sin(particle.lat*math.pi/180) #coriolis parameter
    ucor=-vf*f
    vcor=uf*f

    # drag force
    udrag=particle.tau_inv*(uf-particle.up)
    vdrag=particle.tau_inv*(vf-particle.vp)

    #advection using the Euler-Cromer algorithm:
    a_lon=(particle.Bterm*(DuDt+ucor)+udrag)*particle.dt
    a_lat=(particle.Bterm*(DvDt+vcor)+vdrag)*particle.dt

    particle.up=particle.up+a_lon*particle.dt
    particle.vp=particle.vp+a_lat*particle.dt

    particle_dlon+=particle.up*particle.dt#+uf*particle.dt
    particle_dlat+=particle.vp*particle.dt#vf*particle.dt#

    # sample t-delta_t already for next timestep as we cannot call t-delta_t at timestep t direclty (because parcels only loads in 2 timesteps of the fieldset)
    (particle.uf_tm, particle.vf_tm) =fieldset.UV[time+particle.dt-fieldset.delta_t, particle.depth, particle.lat+particle_dlat, particle.lon+particle_dlon]

    #testing sampling
    #particle.DuDt=DuDt
    #particle.dudt=dudt


def deleteParticle(particle, fieldset, time):
    """ Kernel for deleting particles if they throw an error other than through the surface
    """
    
    if particle.state >= 50:
        #print('delete')
        particle.delete()

    
    



# def MR_AdvectionEE_3D(particle,fieldset,time):

# def MR_AdvectionRK4_2D(particle,fieldset,time):

# def MR_AdvectionRK4_3D(particle,fieldset,time):
