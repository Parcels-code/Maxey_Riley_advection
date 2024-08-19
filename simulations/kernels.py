from parcels import JITParticle, Field, FieldSet, Variable
from datetime import datetime, timedelta
import math
import numpy as np
from operator import attrgetter

class InertialParticle(JITParticle): # particle that is avected according to the MR equation
    ## Terms needed for MR advection:
    tau_inv = Variable('tau_inv',dtype=np.float32, to_write='once',initial=1000.) 
    uf_tm = Variable('uf_tm',dtype=np.float32, to_write=False,initial=0.) #needed because parcels loads in only 2 timesteps of the velocity field
    vf_tm = Variable('vf_tm',dtype=np.float32, to_write=False,initial=0.) #needed because parcels loads in only 2 timesteps of the velocity field
    up= Variable('up',dtype=np.float32, to_write=False, initial=0.) #better to sample with initial velocity field?
    vp= Variable('vp',dtype=np.float32, to_write=False,initial=0.)
    Bterm  = Variable('Bterm',dtype=np.float32, to_write='once',initial=1.)

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
    """
    # read in velocity at location of particle
    (uf,vf)= fieldset.UV[time, particle.depth, particle.lat, particle.lon]

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
    a_lon=(particle.Bterm*(DuDt+ucor)+udrag)
    a_lat=(particle.Bterm*(DvDt+vcor)+vdrag)
    # print(DuDt)
    # print(udrag)
    # print(' ')
    particle.up=particle.up+a_lon*particle.dt
    particle.vp=particle.vp+a_lat*particle.dt

    particle_dlon+=particle.up*particle.dt#+uf*particle.dt
    particle_dlat+=particle.vp*particle.dt#vf*particle.dt#

    # sample t-delta_t already for next timestep as we cannot call t-delta_t at timestep t direclty (because parcels only loads in 2 timesteps of the fieldset)
    (particle.uf_tm, particle.vf_tm) =fieldset.UV[time+particle.dt-fieldset.delta_t, particle.depth, particle.lat+particle_dlat, particle.lon+particle_dlon]

    #testing sampling
    #particle.DuDt=DuDt
    #particle.dudt=dudt


def MRAdvectionRK42D(particle,fieldset,time): 
    """
    Advection of particles using Maxey-Riley equation in 2D
    without Basset history term and Faxen corrections 
    without sinking or floating force. the equation is numerically
    integrated using the 4th order runge kutta scheme for a
    nonoverdamped equation. 

    dependencies:
    the particle should contain up, vp (velocity particle
    at timestep t),tau_inv is 1/stokes time and 
    Bterm=3/(1+2B) with B the buoyancy of the particle. 
    The fieldset should contain constants Omega_earth (angular 
    velocity earth), delta_x, delta_y (stepping for gradients)
    we use delta_t = 0.5*dt otherwhise is does not work
    and we appromate the derivative at t (1rst step rk4) with a
    forward time derivative and the derivative at t+delta_t
    (4th step rk4) with a backward time derivative.
    
    """
      
    norm_deltax=1.0/(2.0*fieldset.delta_x)
    norm_deltay=1.0/(2.0*fieldset.delta_y) 
    norm_deltat=1.0/(1.0*particle.dt)

    ############ RK4 STEP 1 #################
    # read in velocity at location of particle
    (uf1,vf1)= fieldset.UV[time, particle.depth, particle.lat, particle.lon]

    # calculate time derivative of fluid field
    (uf_tp1, vf_tp1) = fieldset.UV[time+particle.dt, particle.depth, particle.lat, particle.lon]
    dudt1=(uf_tp1-uf1)*norm_deltat
    dvdt1=(vf_tp1-vf1)*norm_deltat
 
    
    # calculate spatial gradients fluid field
    (u_dxm1,v_dxm1)=fieldset.UV[time, particle.depth, particle.lat, particle.lon-fieldset.delta_x]
    (u_dxp1,v_dxp1)=fieldset.UV[time, particle.depth, particle.lat, particle.lon+fieldset.delta_x]
    (u_dym1,v_dym1)=fieldset.UV[time, particle.depth, particle.lat-fieldset.delta_y, particle.lon]
    (u_dyp1,v_dyp1)=fieldset.UV[time, particle.depth, particle.lat+fieldset.delta_y, particle.lon]
    dudx1=(u_dxp1-u_dxm1)*norm_deltax
    dudy1=(u_dyp1-u_dym1)*norm_deltay
    dvdx1=(v_dxp1-v_dxm1)*norm_deltax
    dvdy1=(v_dyp1-v_dym1)*norm_deltay

    # use time derivative and spatial gradients to caluclate material derivative fluid
    DuDt1=dudt1+uf1*dudx1+vf1*dudy1
    DvDt1=dvdt1+uf1*dvdx1+vf1*dvdy1

    # coriolis force
    f1=0#2*fieldset.Omega_earth*math.sin(particle.lat*math.pi/180) #coriolis parameter
    ucor1=-vf1*f1
    vcor1=uf1*f1

    # drag force
    udrag1=particle.tau_inv*(uf1-particle.up)
    vdrag1=particle.tau_inv*(vf1-particle.vp)

    # acceleration 
    a_lon1=(particle.Bterm*(DuDt1+ucor1)+udrag1)
    a_lat1=(particle.Bterm*(DvDt1+vcor1)+vdrag1)

    # calculate RK4 coefficients
    u1=a_lon1*particle.dt
    v1=a_lat1*particle.dt
    lon1=particle.up*particle.dt
    lat1=particle.vp*particle.dt

    ############ RK4 STEP 2 #################
    # read in velocity at location of particle
    (uf2,vf2)= fieldset.UV[time+0.5*particle.dt, particle.depth, particle.lat+0.5*lat1, particle.lon+0.5*lon1]

    # calculate time derivative of fluid field
    (uf_tp2, vf_tp2) = fieldset.UV[time+particle.dt, particle.depth, particle.lat+0.5*lat1, particle.lon+0.5*lon1]
    (uf_tm2, vf_tm2) = fieldset.UV[time, particle.depth, particle.lat+0.5*lat1, particle.lon+0.5*lon1]
    dudt2=(uf_tp2-uf_tm2)*norm_deltat
    dvdt2=(vf_tp2-vf_tm2)*norm_deltat
    
    # calculate spatial gradients fluid field
    (u_dxm2,v_dxm2)=fieldset.UV[time+0.5*particle.dt, particle.depth, particle.lat+0.5*lat1, particle.lon+0.5*lon1-fieldset.delta_x]
    (u_dxp2,v_dxp2)=fieldset.UV[time+0.5*particle.dt, particle.depth, particle.lat+0.5*lat1, particle.lon+0.5*lon1+fieldset.delta_x]
    (u_dym2,v_dym2)=fieldset.UV[time+0.5*particle.dt, particle.depth, particle.lat+0.5*lat1-fieldset.delta_y, particle.lon+0.5*lon1]
    (u_dyp2,v_dyp2)=fieldset.UV[time+0.5*particle.dt, particle.depth, particle.lat+0.5*lat1+fieldset.delta_y, particle.lon+0.5*lon1]
    dudx2=(u_dxp2-u_dxm2)*norm_deltax
    dudy2=(u_dyp2-u_dym2)*norm_deltay
    dvdx2=(v_dxp2-v_dxm2)*norm_deltax
    dvdy2=(v_dyp2-v_dym2)*norm_deltay

    # use time derivative and spatial gradients to caluclate material derivative fluid
    DuDt2=dudt2+uf2*dudx2+vf2*dudy2
    DvDt2=dvdt2+uf2*dvdx2+vf2*dvdy2

    # coriolis force
    f2=0#2*fieldset.Omega_earth*math.sin(particle.lat+0.5*lat1*math.pi/180) #coriolis parameter
    ucor2=-vf2*f2
    vcor2=uf2*f2

    # drag force
    udrag2=particle.tau_inv*(uf2-(particle.up+0.5*u1))
    vdrag2=particle.tau_inv*(vf2-(particle.vp+0.5*v1))

    # acceleration 
    a_lon2=particle.Bterm*(DuDt2+ucor2)+udrag2
    a_lat2=particle.Bterm*(DvDt2+vcor2)+vdrag2

    # calculate RK4 coefficients
    u2=a_lon2*particle.dt
    v2=a_lat2*particle.dt
    lon2=(particle.up+0.5*u1)*particle.dt
    lat2=(particle.vp+0.5*v1)*particle.dt

    ############ RK4 STEP 3 #################
    (uf3,vf3)= fieldset.UV[time+0.5*particle.dt, particle.depth, particle.lat+0.5*lat2, particle.lon+0.5*lon2]

    # calculate time derivative of fluid field
    (uf_tp3, vf_tp3) = fieldset.UV[time+particle.dt, particle.depth, particle.lat+0.5*lat2, particle.lon+0.5*lon2]
    (uf_tm3, vf_tm3) = fieldset.UV[time, particle.depth, particle.lat+0.5*lat2, particle.lon+0.5*lon2]
    dudt3=(uf_tp3-uf_tm3)*norm_deltat
    dvdt3=(vf_tp3-vf_tm3)*norm_deltat
    
    # calculate spatial gradients fluid field
    (u_dxm3,v_dxm3)=fieldset.UV[time+0.5*particle.dt, particle.depth, particle.lat+0.5*lat2, particle.lon+0.5*lon2-fieldset.delta_x]
    (u_dxp3,v_dxp3)=fieldset.UV[time+0.5*particle.dt, particle.depth, particle.lat+0.5*lat2, particle.lon+0.5*lon2+fieldset.delta_x]
    (u_dym3,v_dym3)=fieldset.UV[time+0.5*particle.dt, particle.depth, particle.lat+0.5*lat2-fieldset.delta_y, particle.lon+0.5*lon2]
    (u_dyp3,v_dyp3)=fieldset.UV[time+0.5*particle.dt, particle.depth, particle.lat+0.5*lat2+fieldset.delta_y, particle.lon+0.5*lon2]
    dudx3=(u_dxp3-u_dxm3)*norm_deltax
    dudy3=(u_dyp3-u_dym3)*norm_deltay
    dvdx3=(v_dxp3-v_dxm3)*norm_deltax
    dvdy3=(v_dyp3-v_dym3)*norm_deltay

    # use time derivative and spatial gradients to caluclate material derivative fluid
    DuDt3=dudt3+uf3*dudx3+vf3*dudy3
    DvDt3=dvdt3+uf3*dvdx3+vf3*dvdy3

    # coriolis force
    f3=0#2*fieldset.Omega_earth*math.sin(particle.lat+0.5*lat2*math.pi/180) #coriolis parameter
    ucor3=-vf3*f3
    vcor3=uf3*f3

    # drag force
    udrag3=particle.tau_inv*(uf3-(particle.up+0.5*u2))
    vdrag3=particle.tau_inv*(vf3-(particle.vp+0.5*v2))

    # acceleration 
    a_lon3=(particle.Bterm*(DuDt3+ucor3)+udrag3)
    a_lat3=(particle.Bterm*(DvDt3+vcor3)+vdrag3)

    # calculate RK4 coefficients
    u3=a_lon3*particle.dt
    v3=a_lat3*particle.dt
    lon3=(particle.up+0.5*u2)*particle.dt
    lat3=(particle.vp+0.5*v2)*particle.dt


    ############ RK4 STEP 4 #################
    (uf4,vf4)= fieldset.UV[time+particle.dt, particle.depth, particle.lat+lat3, particle.lon+lon3]

    # calculate time derivative of fluid field
    (uf_tp4, vf_tp4) = fieldset.UV[time+particle.dt, particle.depth, particle.lat+lat3, particle.lon+lon3]
    (uf_tm4, vf_tm4) = fieldset.UV[time, particle.depth, particle.lat+lat3, particle.lon+lon3]
    dudt4=(uf_tp4-uf_tm4)*norm_deltat
    dvdt4=(vf_tp4-vf_tm4)*norm_deltat
    
    # calculate spatial gradients fluid field
    (u_dxm4,v_dxm4)=fieldset.UV[time+particle.dt, particle.depth, particle.lat+lat3, particle.lon+lon3-fieldset.delta_x]
    (u_dxp4,v_dxp4)=fieldset.UV[time+particle.dt, particle.depth, particle.lat+lat3, particle.lon+lon3+fieldset.delta_x]
    (u_dym4,v_dym4)=fieldset.UV[time+particle.dt, particle.depth, particle.lat+lat3-fieldset.delta_y, particle.lon+lon3]
    (u_dyp4,v_dyp4)=fieldset.UV[time+particle.dt, particle.depth, particle.lat+lat3+fieldset.delta_y, particle.lon+lon3]
    dudx4=(u_dxp4-u_dxm4)*norm_deltax
    dudy4=(u_dyp4-u_dym4)*norm_deltay
    dvdx4=(v_dxp4-v_dxm4)*norm_deltax
    dvdy4=(v_dyp4-v_dym4)*norm_deltay

    # use time derivative and spatial gradients to caluclate material derivative fluid
    DuDt4=dudt4+uf4*dudx4+vf4*dudy4
    DvDt4=dvdt4+uf4*dvdx4+vf4*dvdy4

    # coriolis force
    f4=0#2*fieldset.Omega_earth*math.sin((particle.lat+lat3)*math.pi/180) #coriolis parameter
    ucor4=-vf4*f4
    vcor4=uf4*f4

    # drag force
    udrag4=particle.tau_inv*(uf4-(particle.up+u3))
    vdrag4=particle.tau_inv*(vf4-(particle.vp+v3))

    # acceleration 
    a_lon4=particle.Bterm*(DuDt4+ucor4)+udrag4
    a_lat4=particle.Bterm*(DvDt4+vcor4)+vdrag4

    # calculate RK4 coefficients
    u4=a_lon4*particle.dt
    v4=a_lat4*particle.dt
    lon4=(particle.up+u3)*particle.dt
    lat4=(particle.vp+v3)*particle.dt


    ############ RK4 INTEGRATION STEP #################
    particle.up+=1.0/6.0*(u1+2*u2+2*u3+u4)
    particle.vp+=1.0/6.0*(v1+2*v2+2*v3+v4)
    particle_dlon+=1.0/6.0*(lon1+2*lon2+2*lon3+lon4)
    particle_dlat+=1.0/6.0*(lat1+2*lat2+2*lat3+lat4)


def deleteParticle(particle, fieldset, time):
    """ Kernel for deleting particles if they throw an error other than through the surface
    """
    
    if particle.state >= 50:
        #print('delete')
        particle.delete()

    
    



# def MR_AdvectionEE_3D(particle,fieldset,time):

# def MR_AdvectionRK4_2D(particle,fieldset,time):

# def MR_AdvectionRK4_3D(particle,fieldset,time):
