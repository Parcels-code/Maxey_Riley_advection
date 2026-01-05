from parcels import JITParticle, Variable
import math
import numpy as np

########## initialize particles #######################
# Class for particle advected with MR equation in 2D
class InertialParticle2D(JITParticle):
    # Terms needed for MR advection:
    B  = Variable('B', dtype=np.float32, to_write='once', initial=1.)
    tau = Variable('tau', dtype=np.float32, to_write='once', initial=1.)
    # velocity particles
    up = Variable('up', dtype=np.float32, to_write=False, initial=0.)
    vp = Variable('vp', dtype=np.float32, to_write=False, initial=0.)


# Class for particle advected with MR equation in 3D
class InertialParticle3D(JITParticle):
    # Terms needed for MR advection:
    tau = Variable('tau', dtype=np.float32, to_write='once', initial=1.)
    B = Variable('B', dtype=np.float32, to_write='once', initial=1.)
    # velocity particles
    up = Variable('up', dtype=np.float32, to_write=False, initial=0.)
    vp = Variable('vp', dtype=np.float32, to_write=False, initial=0.)


def InitializeParticles2D_MRSM(particle, fieldset, time):
    """Kernel for initializing the velocity of the intertial particles in 2D
    to the value of the fluid at the location of the particle.
    """
    Bterm_tau = (2 * (1. - particle.B) / (1. + 2. * particle.B)) * particle.tau

    norm_deltax = 1.0 / (2.0 * fieldset.delta_x)
    norm_deltay = 1.0 / (2.0 * fieldset.delta_y)
    norm_deltat = 1.0 / (1.0 * particle.dt)
    
    # RK4 STEP 1
    # fluid field velocity at location of particle
    (uf1, vf1) = fieldset.UV[time, particle.depth,
                             particle.lat, particle.lon]

    # calculate time derivative of fluid field
    (uf_tp1, vf_tp1) = fieldset.UV[time+particle.dt,
                                   particle.depth, particle.lat, particle.lon]
    (uf_tm1, vf_tm1) = fieldset.UV[time, particle.depth,
                                   particle.lat, particle.lon]
    dudt1 = (uf_tp1 - uf_tm1) * norm_deltat
    dvdt1 = (vf_tp1 - vf_tm1) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm1, v_dxm1) = fieldset.UV[time, particle.depth,
                                particle.lat, particle.lon-fieldset.delta_x]
    (u_dxp1, v_dxp1) = fieldset.UV[time, particle.depth,
                                particle.lat, particle.lon+fieldset.delta_x]
    (u_dym1, v_dym1) = fieldset.UV[time, particle.depth,
                                particle.lat-fieldset.delta_y, particle.lon]
    (u_dyp1, v_dyp1) = fieldset.UV[time, particle.depth,
                                particle.lat+fieldset.delta_y, particle.lon]
    dudx1 = (u_dxp1 - u_dxm1) * norm_deltax
    dudy1 = (u_dyp1 - u_dym1) * norm_deltay
    dvdx1 = (v_dxp1 - v_dxm1) * norm_deltax
    dvdy1 = (v_dyp1 - v_dym1) * norm_deltay

    # caluclate material derivative fluid
    DuDt1 = dudt1 + uf1 * dudx1 + vf1 * dudy1
    DvDt1 = dvdt1 + uf1 * dvdx1 + vf1 * dvdy1

    # coriolis force
    f1 = 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor1 = -vf1 * f1
    vcor1 = uf1 * f1

    particle.up = uf1 + Bterm_tau * (DuDt1 + ucor1)
    particle.vp = vf1 + Bterm_tau * (DvDt1 + vcor1)


def InitializeParticles2D(particle, fieldset, time):
    """Kernel for initializing the velocity of the intertial particles in 2D
    to the value of the fluid at the location of the particle.
    """
    # u, v = fieldset.UV[particle]
    (u, v) = fieldset.UV[time, particle.depth,
                             particle.lat, particle.lon]
    particle.up = u
    particle.vp = v


def InitializeParticles3D(particle, fieldset, time):
    """Kernel for initializing the velocity of the intertial particles in 3D
    to the value of the fluid at the location of the particle for x/y (lon/lat)
    in the z direction the velocity is given by the fluid velocity + the stokes
    settling velocity (w0).

    Dependencies:
    - g, gravitational acceleration (fieldset constant)
    - tau, stokes relaxation time (particle variable)
    - B, buoyancy particle (particle variable)
    """
    u, v, w = fieldset.UVW[particle]
    w0 = (2 * (1 - particle.B) / (1 + 2 * particle.B)
          * fieldset.g * particle.tau)
    particle.up = u
    particle.vp = v
    particle.wp = w + w0


#################### delete particles #################
def deleteParticle(particle, fieldset, time):
    """ Kernel for deleting particles if they throw an error other than through
    the surface
    """
    if particle.state >= 50:
        particle.delete()


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


def too_close_to_edge(particle, fieldset, time):
    """
    Kernel to delete particles too close to the edge to be able to calculate
    the derivative of the fluid at that location using finite differences

    Dependencies:
    - lon_min,lon_max, lat_min, lat_max, boundaries of domain (fieldset
      constants)
    - delta_x, delta_y, stepsize used for calculating derivatives (fieldset
      constants)
    """
    if(math.fabs(particle.lon - fieldset.lon_min) < 2 * fieldset.delta_x):
        particle.delete()
    if(math.fabs(particle.lat - fieldset.lat_min) < 2 * fieldset.delta_y):
        particle.delete()
    if(math.fabs(fieldset.lon_max - particle.lon) < 2 * fieldset.delta_x):
        particle.delete()
    if(math.fabs(fieldset.lat_max - particle.lat) < 2 * fieldset.delta_y):
        particle.delete()

################## anti beaching ####################
def set_displacement(particle, fieldset, time):
    """ 
    Kernel to check if particles are close to the coast and if they are
    set the particles up for displacement away from the coast. The actual
    displacement of th eparticles is doen int the kernel "displace"

    Dependencies: 
    fieldset with distance to shore (distance2shore)
    fieldste with displacement field away from shore (dispU, dispV)
    """
    particle.d2s = fieldset.distance2shore[
        time, particle.depth, particle.lat, particle.lon
    ]
    if particle.d2s < 0.5:
        dispUab = fieldset.dispU[time, particle.depth, particle.lat, particle.lon]
        dispVab = fieldset.dispV[time, particle.depth, particle.lat, particle.lon]
        particle.dU = dispUab
        particle.dV = dispVab
    else:
        particle.dU = 0.0
        particle.dV = 0.0


def displace(particle, fieldset, time):
    """
    Displaces particles that are close to the coast away from the coast over
    distance DU*dt where DU is calculated based on the distance form the coast
    in the kernel set_displacement 

    Dependencies:
    - Displacement velocity away from coast (dU, dV) calculated in
    kernel set_displacment
    """
    if particle.d2s < 0.5:
        particle_dlon += particle.dU * particle.dt
        particle_dlat += particle.dV * particle.dt


############## function to measure properties flow at particle level ##########
def measure_vorticity(particle, fieldset, time):
    """
    Kernel to measure the vorticity of the fluid at the location 
    of the particle

    Dependencies:
    - length used for calculating derivatives (delta_x, delta_y)
    """
    u_dxm, v_dxm = fieldset.UV[time, particle.depth,
                                 particle.lat, particle.lon - fieldset.delta_x]
    u_dxp, v_dxp = fieldset.UV[time, particle.depth,
                                 particle.lat, particle.lon + fieldset.delta_x]
    u_dym, v_dym = fieldset.UV[time, particle.depth,
                                 particle.lat - fieldset.delta_y, particle.lon]
    u_dyp, v_dyp = fieldset.UV[time, particle.depth,
                                 particle.lat + fieldset.delta_y, particle.lon]
    dvdx = (v_dxp - v_dxm) / (2 * fieldset.delta_x)
    dudy = (u_dyp - u_dym) / (2 * fieldset.delta_y)
    particle.vorticity = dvdx - dudy

def measure_fluid_velocity(particle, fieldset, time):
    """
    Kernel to measure the velocity (2D) of the fluid at the location 
    of the particle
    """
    uf, vf = fieldset.UV.eval(time, particle.depth, particle.lat,
                              particle.lon, applyConversion=False)
    
    particle.ufluid = uf
    particle.vfluid = vf


def measure_particle_velocity(particle, fieldset, time):
    """
    Kernel to measure the velocity (2D) of the fluid at the location 
    of the particle
    """    
    particle.uparticle = (particle.up)* fieldset.Rearth * math.cos(particle.lat * math.pi / 180.) * math.pi / 180.
    particle.vparticle = (particle.vp) * fieldset.Rearth * math.pi / 180. 


def measure_slip_velocity(particle, fieldset, time):
    """f
    Kernel to measure the slip velocity (2D) of the fluid at the location 
    of the particle
    """
    uf, vf = fieldset.UV[time, particle.depth, particle.lat,
                              particle.lon]



    particle.uslip= (particle.up - uf)* fieldset.Rearth * math.cos(particle.lat * math.pi / 180.) * math.pi / 180.
    particle.vslip = (particle.vp - vf) * fieldset.Rearth * math.pi / 180. 

    
def measure_coriolis_force(particle, fieldset, time):
    """f
    Kernel to measure the coriolis force (2D) of the fluid at the location 
    of the particle
    """
    Bterm = 3. / (1. + 2. * particle.B)
    Bterm2 = 2 * (1 - particle.B) / (1 + 2 * particle.B) 
    uf, vf = fieldset.UV[time, particle.depth, particle.lat,
                              particle.lon]

    # coriolis force
    f = 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor = -(vf - particle.vp) * f
    vcor = (uf - particle.up) * f
    upcor = -particle.vp * f
    vpcor = particle.up * f
    fcor_u = Bterm * ucor+ Bterm2 * upcor
    fcor_v = Bterm * vcor + Bterm2 * vpcor
    particle.fcor_lon = fcor_u * fieldset.Rearth * math.cos(particle.lat * math.pi / 180.) * math.pi / 180.
    particle.fcor_lat = fcor_v * fieldset.Rearth * math.pi / 180. 

def measure_acceleration_force(particle, fieldset, time):
    """
    Kernel to measure the acceleration force (2D) of the fluid at the location 
    of the particle
    """
    Bterm = 3. / (1. + 2. * particle.B)
    norm_deltax = 1.0 / (2.0 * fieldset.delta_x)
    norm_deltay = 1.0 / (2.0 * fieldset.delta_y)
    norm_deltat = 1.0 / (1.0 * particle.dt)


    # read in velocity at location of particle
    (uf1, vf1) = fieldset.UV[time,
                             particle.depth, particle.lat, particle.lon]


    # calculate time derivative of fluid field
    (uf_tp1, vf_tp1) = fieldset.UV[time+particle.dt,
                                   particle.depth, particle.lat, particle.lon]
    (uf_tm1, vf_tm1) = fieldset.UV[time,
                                   particle.depth, particle.lat, particle.lon]
    dudt1 = (uf_tp1 - uf_tm1) * norm_deltat
    dvdt1 = (vf_tp1 - vf_tm1) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm1, v_dxm1) = fieldset.UV[time, particle.depth, particle.lat,
                                   particle.lon - fieldset.delta_x]
    (u_dxp1, v_dxp1) = fieldset.UV[time, particle.depth, particle.lat,
                                   particle.lon + fieldset.delta_x]
    (u_dym1, v_dym1) = fieldset.UV[time, particle.depth, particle.lat
                                   - fieldset.delta_y, particle.lon]
    (u_dyp1, v_dyp1) = fieldset.UV[time, particle.depth, particle.lat
                                   + fieldset.delta_y, particle.lon]
    dudx1 = (u_dxp1 - u_dxm1) * norm_deltax
    dudy1 = (u_dyp1 - u_dym1) * norm_deltay
    dvdx1 = (v_dxp1 - v_dxm1) * norm_deltax
    dvdy1 = (v_dyp1 - v_dym1) * norm_deltay

    # caluclate material derivative fluid
    DuDt1 = dudt1 + uf1 * dudx1 + vf1 * dudy1
    DvDt1 = dvdt1 + uf1 * dvdx1 + vf1 * dvdy1

     # acceleration force
    facc_lon = Bterm * DuDt1 
    facc_lat = Bterm * DvDt1 
    particle.facc_lon = facc_lon * fieldset.Rearth * math.cos(particle.lat * math.pi / 180.) * math.pi / 180.
    particle.facc_lat = facc_lat * fieldset.Rearth * math.pi / 180. 


def measure_slip_velocity_SM(particle, fieldset, time):
    """f
    Kernel to measure the slip velocity (2D) of the fluid at the location 
    of the particle
    """

    particle.uf = particle.up * fieldset.Rearth * math.cos(particle.lat * math.pi / 180.) * math.pi / 180.
    particle.vf = particle.vp * fieldset.Rearth * math.pi / 180. 




##### Maxey-Riley-Gatinol advection ###################

def MRAdvectionRK4_2D_drag_Rep(particle, fieldset, time):
    """
    Advection of particles using Maxey-Riley equation in 2D without Basset
    history term and Faxen corrections without sinking or floating force.
    The equation is numerically integrated using the 4th order runge kutta
    scheme for a 2nd order ODE equation. We appromate the time derivative at t 
    (1rst step rk4) with a forward finite difference and the time derivative
    at t+delta_t (4th step rk4) with a backward finite difference.
    We use the a fluid drag that depends on the particle reynolds number.
    We use the equation given by morrision 2013 which is valid up to Re_p= 10^6. 


    dependencies:
    - up, vp, particle velocity (particle variables)
    - B, buoyancy particle (particle variable)
    - Omega_earth, angular velocity earth (fieldset constant)
    - nu, kinematic velocity (fieldset constant) [m/s^2]
    - d, diameter particle, (particle constant)  [m]
    - delta_x, delta_y, delta_t step for finite difference method gradients 
      (fieldset constants)
    """ 
    Bterm = 3. / (1. + 2. * particle.B)
    Bterm2 = 2 * (1 - particle.B) / (1 + 2 * particle.B) 
    norm_deltax = 1.0 / (2.0 * fieldset.delta_x)
    norm_deltay = 1.0 / (2.0 * fieldset.delta_y)
    norm_deltat = 1.0 / (1.0 * particle.dt)


    #VELOCITY DEPENDEND DRAG STOKES TIME
    #get velocity flow and particle in m/s
    (uf, vf) = fieldset.UV[time, particle.depth, particle.lat,
                              particle.lon]
    
    uslip = (particle.up -uf) * fieldset.Rearth * math.cos(particle.lat * math.pi /180) *  math.pi / 180.
    vslip = (particle.vp -vf) * fieldset.Rearth * math.pi / 180
    
    if(fieldset.save_slip_velocity == True):
        particle.uslip = uslip
        particle.vslip = vslip
    #calculate Reynolds number
    Rep = math.sqrt((uslip)**2 +(vslip)**2) * particle.diameter / (fieldset.nu)
    if(Rep > 5000): # to make simulation stable
        Rep = 5000
    #save Reynolds number
    # particle.Rep = Rep
    #calulate correction factor
    f_REp = 1 + Rep / (4. * (1 + math.sqrt(Rep))) + Rep / 60.

    # inverse stokes time based on correction factor
    tau_inv = 36 * fieldset.nu * f_REp /( (1. + 2. * particle.B) * particle.diameter**2)

    # RK4 STEP 1
    # read in velocity at location of particle
    (uf1, vf1) = fieldset.UV[time,
                             particle.depth, particle.lat, particle.lon]

    # velocity particle at current step
    up1 = particle.up
    vp1 = particle.vp

    

    # calculate time derivative of fluid field
    (uf_tp1, vf_tp1) = fieldset.UV[time+particle.dt,
                                   particle.depth, particle.lat, particle.lon]
    (uf_tm1, vf_tm1) = fieldset.UV[time,
                                   particle.depth, particle.lat, particle.lon]
    dudt1 = (uf_tp1 - uf_tm1) * norm_deltat
    dvdt1 = (vf_tp1 - vf_tm1) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm1, v_dxm1) = fieldset.UV[time, particle.depth, particle.lat,
                                   particle.lon - fieldset.delta_x]
    (u_dxp1, v_dxp1) = fieldset.UV[time, particle.depth, particle.lat,
                                   particle.lon + fieldset.delta_x]
    (u_dym1, v_dym1) = fieldset.UV[time, particle.depth, particle.lat
                                   - fieldset.delta_y, particle.lon]
    (u_dyp1, v_dyp1) = fieldset.UV[time, particle.depth, particle.lat
                                   + fieldset.delta_y, particle.lon]
    dudx1 = (u_dxp1 - u_dxm1) * norm_deltax
    dudy1 = (u_dyp1 - u_dym1) * norm_deltay
    dvdx1 = (v_dxp1 - v_dxm1) * norm_deltax
    dvdy1 = (v_dyp1 - v_dym1) * norm_deltay

    # caluclate material derivative fluid
    DuDt1 =   ( dudt1 + uf1 * dudx1 + vf1 * dudy1 )
    DvDt1 =   ( dvdt1 + uf1 * dvdx1 + vf1 * dvdy1 )

    
    # coriolis force
    f1 = 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor1 = -(vf1 - vp1) * f1
    vcor1 = (uf1 - up1) * f1
    upcor1 = -vp1 * f1
    vpcor1 = up1 * f1

    #lift force    
    omega1 = dvdx1 - dudy1
    ulift1 = -(vf1 - vp1) * omega1
    vlift1 = (uf1 - up1) * omega1

    # drag force
    udrag1 = tau_inv * (uf1 - up1)
    vdrag1 = tau_inv * (vf1 - vp1)

    # acceleration
    a_lon1 = Bterm * (DuDt1 + ucor1 + ulift1) + Bterm2 * upcor1 + udrag1
    a_lat1 = Bterm * (DvDt1 + vcor1 + vlift1) + Bterm2 * vpcor1 + vdrag1

    # lon, lat for next step
    lon1 = particle.lon + 0.5 * up1 * particle.dt
    lat1 = particle.lat + 0.5 * vp1 * particle.dt
    time1 = time + 0.5 * particle.dt

    # RK4 STEP 2
    # velocity particle at current step
    up2 = particle.up + 0.5 * a_lon1 * particle.dt
    vp2 = particle.vp + 0.5 * a_lat1 * particle.dt

    # read in velocity at location of particle
    (uf2, vf2) = fieldset.UV[time1, particle.depth, lat1, lon1]

    # calculate time derivative of fluid field
    (uf_tp2, vf_tp2) = fieldset.UV[time + particle.dt,
                                   particle.depth, lat1, lon1]
    (uf_tm2, vf_tm2) = fieldset.UV[time, particle.depth, lat1, lon1]
    dudt2 = (uf_tp2 - uf_tm2) * norm_deltat
    dvdt2 = (vf_tp2 - vf_tm2) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm2, v_dxm2) = fieldset.UV[time1, particle.depth,
                                   lat1, lon1 - fieldset.delta_x]
    (u_dxp2, v_dxp2) = fieldset.UV[time1, particle.depth,
                                   lat1, lon1 + fieldset.delta_x]
    (u_dym2, v_dym2) = fieldset.UV[time1, particle.depth,
                                   lat1 - fieldset.delta_y, lon1]
    (u_dyp2, v_dyp2) = fieldset.UV[time1, particle.depth,
                                   lat1 + fieldset.delta_y, lon1]
    dudx2 = (u_dxp2 - u_dxm2) * norm_deltax
    dudy2 = (u_dyp2 - u_dym2) * norm_deltay
    dvdx2 = (v_dxp2 - v_dxm2) * norm_deltax
    dvdy2 = (v_dyp2 - v_dym2) * norm_deltay

    # caluclate material derivative fluid
    DuDt2 =   ( dudt2 + uf2 * dudx2 + vf2 * dudy2 )
    DvDt2 =   (dvdt2 + uf2 * dvdx2 + vf2 * dvdy2 ) 

    # coriolis force
    f2 = 2 * fieldset.Omega_earth * math.sin(lat1 * math.pi / 180.)
    ucor2 = -(vf2 - vp2) * f2
    vcor2 = (uf2 - up2) * f2
    upcor2 = -vp2 * f2
    vpcor2 = up2 * f2

    #lift force    
    omega2 = dvdx2 - dudy2
    ulift2 = -(vf2 - vp2) * omega2
    vlift2 = (uf2 - up2) * omega2

    # drag force
    udrag2 = tau_inv * (uf2 - up2)
    vdrag2 = tau_inv * (vf2 - vp2)

    # acceleration
    a_lon2 = Bterm * (DuDt2 + ucor2 + ulift2) + Bterm2 * upcor2 + udrag2
    a_lat2 = Bterm * (DvDt2 + vcor2 + vlift2) + Bterm2 * vpcor2 + vdrag2

    # lon, lat for next step
    lon2 = particle.lon + 0.5 * up2 * particle.dt
    lat2 = particle.lat + 0.5 * vp2 * particle.dt
    time2 = time + 0.5 * particle.dt

    # RK4 STEP 3
    # velocity particle at current step
    up3 = particle.up + 0.5 * a_lon2 * particle.dt
    vp3 = particle.vp + 0.5 * a_lat2 * particle.dt

    # read in velocity at location of particle
    (uf3, vf3) = fieldset.UV[time2, particle.depth, lat2, lon2]

    # calculate time derivative of fluid field
    (uf_tp3, vf_tp3) = fieldset.UV[time + particle.dt,
                                   particle.depth, lat2, lon2]
    (uf_tm3, vf_tm3) = fieldset.UV[time,
                                   particle.depth, lat2, lon2]
    dudt3 = (uf_tp3 - uf_tm3) * norm_deltat
    dvdt3 = (vf_tp3 - vf_tm3) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm3, v_dxm3) = fieldset.UV[time2, particle.depth,
                                   lat2, lon2-fieldset.delta_x]
    (u_dxp3, v_dxp3) = fieldset.UV[time2, particle.depth,
                                   lat2, lon2 + fieldset.delta_x]
    (u_dym3, v_dym3) = fieldset.UV[time2, particle.depth,
                                   lat2 - fieldset.delta_y, lon2]
    (u_dyp3, v_dyp3) = fieldset.UV[time2, particle.depth,
                                   lat2 + fieldset.delta_y, lon2]
    dudx3 = (u_dxp3 - u_dxm3) * norm_deltax
    dudy3 = (u_dyp3 - u_dym3) * norm_deltay
    dvdx3 = (v_dxp3 - v_dxm3) * norm_deltax
    dvdy3 = (v_dyp3 - v_dym3) * norm_deltay

    # caluclate material derivative fluid
    DuDt3 =   (dudt3 + uf3 * dudx3 + vf3 * dudy3)
    DvDt3 =   (  dvdt3 + uf3 * dvdx3 + vf3 * dvdy3)

    # coriolis force
    f3 = 2 * fieldset.Omega_earth * math.sin(lat2 * math.pi / 180.)
    ucor3 = -(vf3 - vp3) * f3
    vcor3 = (uf3 - up3) * f3
    upcor3 = -vp3 * f3
    vpcor3 = up3 * f3

    #lift force    
    omega3 = dvdx3 - dudy3
    ulift3 = -(vf3 - vp3) * omega3
    vlift3 = (uf3 - up3) * omega3

    # drag force
    udrag3 = tau_inv * (uf3 - up3)
    vdrag3 = tau_inv * (vf3 - vp3)

    # acceleration
    a_lon3 = Bterm * (DuDt3 + ucor3 + ulift3) + Bterm2 * upcor3 + udrag3
    a_lat3 = Bterm * (DvDt3 + vcor3 + vlift3) + Bterm2 * vpcor3 + vdrag3

    # lon, lat for next step
    lon3 = particle.lon + up3 * particle.dt
    lat3 = particle.lat + vp3 * particle.dt
    time3 = time + particle.dt

    # RK4 STEP 4
    # velocity particle at current step
    up4 = particle.up + a_lon3 * particle.dt
    vp4 = particle.vp + a_lat3 * particle.dt

    # read in velocity at location of particle
    (uf4, vf4) = fieldset.UV[time3, particle.depth, lat3, lon3]

    # calculate time derivative of fluid field
    (uf_tp4, vf_tp4) = fieldset.UV[time + particle.dt,
                                   particle.depth, lat3, lon3]
    (uf_tm4, vf_tm4) = fieldset.UV[time, particle.depth, lat3, lon3]
    dudt4 = (uf_tp4 - uf_tm4) * norm_deltat
    dvdt4 = (vf_tp4 - vf_tm4) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm4, v_dxm4) = fieldset.UV[time3, particle.depth,
                                   lat3, lon3 - fieldset.delta_x]
    (u_dxp4, v_dxp4) = fieldset.UV[time3, particle.depth,
                                   lat3, lon3 + fieldset.delta_x]
    (u_dym4, v_dym4) = fieldset.UV[time3, particle.depth,
                                   lat3 - fieldset.delta_y, lon3]
    (u_dyp4, v_dyp4) = fieldset.UV[time3, particle.depth,
                                   lat3 + fieldset.delta_y, lon3]
    dudx4 = (u_dxp4 - u_dxm4) * norm_deltax
    dudy4 = (u_dyp4 - u_dym4) * norm_deltay
    dvdx4 = (v_dxp4 - v_dxm4) * norm_deltax
    dvdy4 = (v_dyp4 - v_dym4) * norm_deltay

    # caluclate material derivative fluid
    DuDt4 =   (dudt4 + uf4 * dudx4 + vf4 * dudy4)
    DvDt4 =   (dvdt4 + uf4 * dvdx4 + vf4 * dvdy4 ) 

    # coriolis force
    f4 =  2 * fieldset.Omega_earth * math.sin(lat3 * math.pi / 180)
    ucor4 = -(vf4 - vp4) * f4
    vcor4 = (uf4 - up4) * f4
    upcor4 = -vp4 * f4
    vpcor4 = up4 * f4

    #lift force    
    omega4 = dvdx4 - dudy4
    ulift4 = -(vf4 - vp4) * omega4
    vlift4 = (uf4 - up4) * omega4

    # drag force
    udrag4 = tau_inv * (uf4-up4)
    vdrag4 = tau_inv * (vf4-vp4)

    # acceleration
    a_lon4 = Bterm * (DuDt4 + ucor4 + ulift4) + Bterm2 * upcor4 + udrag4
    a_lat4 = Bterm * (DvDt4 + vcor4 + vlift4) + Bterm2 * vpcor4 + vdrag4

    # RK4 INTEGRATION STEP
    particle.up += (a_lon1 + 2 * a_lon2
                    + 2 * a_lon3 + a_lon4) * particle.dt / 6.0
    particle.vp += (a_lat1 + 2 * a_lat2
                    + 2 * a_lat3 + a_lat4) * particle.dt / 6.0
    particle_dlon += (up1 + 2 * up2 + 2 * up3 + up4) * particle.dt / 6.0
    particle_dlat += (vp1 + 2 * vp2 + 2 * vp3 + vp4) * particle.dt / 6.0


def MRAdvectionRK4_2D_drag_Rep_constant(particle, fieldset, time):
    """
    Advection of particles using Maxey-Riley equation in 2D without Basset
    history term and Faxen corrections without sinking or floating force.
    The equation is numerically integrated using the 4th order runge kutta
    scheme for a 2nd order ODE equation. We appromate the time derivative at t 
    (1rst step rk4) with a forward finite difference and the time derivative
    at t+delta_t (4th step rk4) with a backward finite difference.
    We use a preset particle reynolds number


    dependencies:
    - up, vp, particle velocity (particle variables)
    - B, buoyancy particle (particle variable)
    - Omega_earth, angular velocity earth (fieldset constant)
    - tau stokes relaxatioin time (particle variable)
    - C_Rep corection factor viscosity for Rep > 1
    - d, diameter particle, (particle constant)  [m]
    - delta_x, delta_y, delta_t step for finite difference method gradients 
      (fieldset constants)
    """ 
    Bterm = 3. / (1. + 2. * particle.B)
    Bterm2 = 2 * (1 - particle.B) / (1 + 2 * particle.B) 
    norm_deltax = 1.0 / (2.0 * fieldset.delta_x)
    norm_deltay = 1.0 / (2.0 * fieldset.delta_y)
    norm_deltat = 1.0 / (1.0 * particle.dt)
    tau_inv = particle.C_Rep / particle.tau

    # RK4 STEP 1
    # read in velocity at location of particle
    # slip velocity
    (uf1, vf1) = fieldset.UV[time,
                             particle.depth, particle.lat, particle.lon]
    
    up1 = particle.up
    vp1 = particle.vp

    uslip = (up1 -uf1) * fieldset.Rearth * math.cos(particle.lat * math.pi /180) *  math.pi / 180.
    vslip = (vp1 -vf1) * fieldset.Rearth * math.pi / 180
    
    if(fieldset.save_slip_velocity == True):
        particle.uslip = uslip
        particle.vslip = vslip
    
   
    
    

    # calculate time derivative of fluid field
    (uf_tp1, vf_tp1) = fieldset.UV[time+particle.dt,
                                   particle.depth, particle.lat, particle.lon]
    (uf_tm1, vf_tm1) = fieldset.UV[time,
                                   particle.depth, particle.lat, particle.lon]
    dudt1 = (uf_tp1 - uf_tm1) * norm_deltat
    dvdt1 = (vf_tp1 - vf_tm1) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm1, v_dxm1) = fieldset.UV[time, particle.depth, particle.lat,
                                   particle.lon - fieldset.delta_x]
    (u_dxp1, v_dxp1) = fieldset.UV[time, particle.depth, particle.lat,
                                   particle.lon + fieldset.delta_x]
    (u_dym1, v_dym1) = fieldset.UV[time, particle.depth, particle.lat
                                   - fieldset.delta_y, particle.lon]
    (u_dyp1, v_dyp1) = fieldset.UV[time, particle.depth, particle.lat
                                   + fieldset.delta_y, particle.lon]
    dudx1 = (u_dxp1 - u_dxm1) * norm_deltax
    dudy1 = (u_dyp1 - u_dym1) * norm_deltay
    dvdx1 = (v_dxp1 - v_dxm1) * norm_deltax
    dvdy1 = (v_dyp1 - v_dym1) * norm_deltay

    # caluclate material derivative fluid
    DuDt1 = dudt1 + uf1 * dudx1 + vf1 * dudy1
    DvDt1 = dvdt1 + uf1 * dvdx1 + vf1 * dvdy1

    #slip velocity
    uslip1 = uf1 - up1
    vslip1 = vf1 - vp1

    # coriolis force
    f1 = 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor1 = -vslip1 * f1
    vcor1 = uslip1 * f1
    upcor1 = -vp1 * f1
    vpcor1 = up1 * f1

    #lift force
    omega1 = dvdx1 - dudy1
    ulift1 = -vslip1 * omega1
    vlift1 = uslip1 * omega1

    # drag force
    udrag1 = tau_inv * uslip1
    vdrag1 = tau_inv * vslip1

    # acceleration
    a_lon1 = Bterm * (DuDt1 + ucor1 + ulift1) + Bterm2 * upcor1 + udrag1
    a_lat1 = Bterm * (DvDt1 + vcor1 + vlift1) + Bterm2 * vpcor1 + vdrag1

    # lon, lat for next step
    lon1 = particle.lon + 0.5 * up1 * particle.dt
    lat1 = particle.lat + 0.5 * vp1 * particle.dt
    time1 = time + 0.5 * particle.dt

    # RK4 STEP 2
    # velocity particle at current step
    up2 = particle.up + 0.5 * a_lon1 * particle.dt
    vp2 = particle.vp + 0.5 * a_lat1 * particle.dt

    # read in velocity at location of particle
    (uf2, vf2) = fieldset.UV[time1, particle.depth, lat1, lon1]

    # calculate time derivative of fluid field
    (uf_tp2, vf_tp2) = fieldset.UV[time + particle.dt,
                                   particle.depth, lat1, lon1]
    (uf_tm2, vf_tm2) = fieldset.UV[time, particle.depth, lat1, lon1]
    dudt2 = (uf_tp2 - uf_tm2) * norm_deltat
    dvdt2 = (vf_tp2 - vf_tm2) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm2, v_dxm2) = fieldset.UV[time1, particle.depth,
                                   lat1, lon1 - fieldset.delta_x]
    (u_dxp2, v_dxp2) = fieldset.UV[time1, particle.depth,
                                   lat1, lon1 + fieldset.delta_x]
    (u_dym2, v_dym2) = fieldset.UV[time1, particle.depth,
                                   lat1 - fieldset.delta_y, lon1]
    (u_dyp2, v_dyp2) = fieldset.UV[time1, particle.depth,
                                   lat1 + fieldset.delta_y, lon1]
    dudx2 = (u_dxp2 - u_dxm2) * norm_deltax
    dudy2 = (u_dyp2 - u_dym2) * norm_deltay
    dvdx2 = (v_dxp2 - v_dxm2) * norm_deltax
    dvdy2 = (v_dyp2 - v_dym2) * norm_deltay

    # caluclate material derivative fluid
    DuDt2 = dudt2 + uf2 * dudx2 + vf2 * dudy2
    DvDt2 = dvdt2 + uf2 * dvdx2 + vf2 * dvdy2

    # slip velocity
    uslip2 = uf2 - up2
    vslip2 = vf2 - vp2
    # coriolis force
    f2 = 2 * fieldset.Omega_earth * math.sin(lat1 * math.pi / 180.)
    ucor2 = -vslip2 * f2
    vcor2 = uslip2 * f2
    upcor2 = -vp2 * f2
    vpcor2 = up2 * f2

    #lift force
    omega2 = dvdx2 - dudy2
    ulift2 = -vslip2 * omega2
    vlift2 = uslip2 * omega2

    # drag force
    udrag2 = tau_inv * uslip2
    vdrag2 = tau_inv * vslip2

    # acceleration
    a_lon2 = Bterm * (DuDt2 + ucor2 + ulift2) + Bterm2 * upcor2 + udrag2
    a_lat2 = Bterm * (DvDt2 + vcor2 + vlift2) + Bterm2 * vpcor2 + vdrag2

    # lon, lat for next step
    lon2 = particle.lon + 0.5 * up2 * particle.dt
    lat2 = particle.lat + 0.5 * vp2 * particle.dt
    time2 = time + 0.5 * particle.dt

    # RK4 STEP 3
    # velocity particle at current step
    up3 = particle.up + 0.5 * a_lon2 * particle.dt
    vp3 = particle.vp + 0.5 * a_lat2 * particle.dt

    # read in velocity at location of particle
    (uf3, vf3) = fieldset.UV[time2, particle.depth, lat2, lon2]

    # calculate time derivative of fluid field
    (uf_tp3, vf_tp3) = fieldset.UV[time + particle.dt,
                                   particle.depth, lat2, lon2]
    (uf_tm3, vf_tm3) = fieldset.UV[time,
                                   particle.depth, lat2, lon2]
    dudt3 = (uf_tp3 - uf_tm3) * norm_deltat
    dvdt3 = (vf_tp3 - vf_tm3) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm3, v_dxm3) = fieldset.UV[time2, particle.depth,
                                   lat2, lon2-fieldset.delta_x]
    (u_dxp3, v_dxp3) = fieldset.UV[time2, particle.depth,
                                   lat2, lon2 + fieldset.delta_x]
    (u_dym3, v_dym3) = fieldset.UV[time2, particle.depth,
                                   lat2 - fieldset.delta_y, lon2]
    (u_dyp3, v_dyp3) = fieldset.UV[time2, particle.depth,
                                   lat2 + fieldset.delta_y, lon2]
    dudx3 = (u_dxp3 - u_dxm3) * norm_deltax
    dudy3 = (u_dyp3 - u_dym3) * norm_deltay
    dvdx3 = (v_dxp3 - v_dxm3) * norm_deltax
    dvdy3 = (v_dyp3 - v_dym3) * norm_deltay

    # caluclate material derivative fluid
    DuDt3 = dudt3 + uf3 * dudx3 + vf3 * dudy3
    DvDt3 = dvdt3 + uf3 * dvdx3 + vf3 * dvdy3

    #slip velocity 
    uslip3 = uf3 - up3
    vslip3 = vf3 - vp3
    
    # coriolis force
    f3 = 2 * fieldset.Omega_earth * math.sin(lat2 * math.pi / 180.)
    ucor3 = -(vslip3) * f3
    vcor3 = (uslip3) * f3
    upcor3 = -vp3 * f3
    vpcor3 = up3 * f3

    #lift force
    omega3 = dvdx3 - dudy3
    ulift3 = -vslip3 * omega3
    vlift3 = uslip3 * omega3

    # drag force
    udrag3 = tau_inv * (uslip3)
    vdrag3 = tau_inv * (vslip3)

    # acceleration
    a_lon3 = Bterm * (DuDt3 + ucor3 + ulift3) + Bterm2 * upcor3 + udrag3
    a_lat3 = Bterm * (DvDt3 + vcor3 + vlift3) + Bterm2 * vpcor3 + vdrag3

    # lon, lat for next step
    lon3 = particle.lon + up3 * particle.dt
    lat3 = particle.lat + vp3 * particle.dt
    time3 = time + particle.dt

    # RK4 STEP 4
    # velocity particle at current step
    up4 = particle.up + a_lon3 * particle.dt
    vp4 = particle.vp + a_lat3 * particle.dt

    # read in velocity at location of particle
    (uf4, vf4) = fieldset.UV[time3, particle.depth, lat3, lon3]

    # calculate time derivative of fluid field
    (uf_tp4, vf_tp4) = fieldset.UV[time + particle.dt,
                                   particle.depth, lat3, lon3]
    (uf_tm4, vf_tm4) = fieldset.UV[time, particle.depth, lat3, lon3]
    dudt4 = (uf_tp4 - uf_tm4) * norm_deltat
    dvdt4 = (vf_tp4 - vf_tm4) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm4, v_dxm4) = fieldset.UV[time3, particle.depth,
                                   lat3, lon3 - fieldset.delta_x]
    (u_dxp4, v_dxp4) = fieldset.UV[time3, particle.depth,
                                   lat3, lon3 + fieldset.delta_x]
    (u_dym4, v_dym4) = fieldset.UV[time3, particle.depth,
                                   lat3 - fieldset.delta_y, lon3]
    (u_dyp4, v_dyp4) = fieldset.UV[time3, particle.depth,
                                   lat3 + fieldset.delta_y, lon3]
    dudx4 = (u_dxp4 - u_dxm4) * norm_deltax
    dudy4 = (u_dyp4 - u_dym4) * norm_deltay
    dvdx4 = (v_dxp4 - v_dxm4) * norm_deltax
    dvdy4 = (v_dyp4 - v_dym4) * norm_deltay

    # caluclate material derivative fluid
    DuDt4 = dudt4 + uf4 * dudx4 + vf4 * dudy4
    DvDt4 = dvdt4 + uf4 * dvdx4 + vf4 * dvdy4

    #slip velocity 
    uslip4 = uf4 - up4
    vslip4 = vf4 - vp4

    # coriolis force
    f4 =  2 * fieldset.Omega_earth * math.sin(lat3 * math.pi / 180)
    ucor4 = -(vslip4) * f4
    vcor4 = (uslip4) * f4
    upcor4 = -vp4 * f4
    vpcor4 = up4 * f4

    #lift force
    omega4 = dvdx4 - dudy4
    ulift4 = -vslip4 * omega4
    vlift4 = uslip4 * omega4

    # drag force
    udrag4 = tau_inv * (uslip4)
    vdrag4 = tau_inv * (vslip4)

    # acceleration
    a_lon4 = Bterm * (DuDt4 + ucor4 + ulift4) + Bterm2 * upcor4 + udrag4
    a_lat4 = Bterm * (DvDt4 + vcor4 + vlift4) + Bterm2 * vpcor4 + vdrag4

    # RK4 INTEGRATION STEP
    particle.up += (a_lon1 + 2 * a_lon2
                    + 2 * a_lon3 + a_lon4) * particle.dt / 6.0
    particle.vp += (a_lat1 + 2 * a_lat2
                    + 2 * a_lat3 + a_lat4) * particle.dt / 6.0
    particle_dlon += (up1 + 2 * up2 + 2 * up3 + up4) * particle.dt / 6.0
    particle_dlat += (vp1 + 2 * vp2 + 2 * vp3 + vp4) * particle.dt / 6.0
 






#  SLOW-MANIFOLD MAXEY-RILEY-GATIGNOL
def MRSMAdvectionRK4_2D_drag_Rep(particle, fieldset, time):
    """
    Advection of particles using the slow manifold approximation of the Maxey-
    Riley equation in 2D without Basset history term and Faxen corrections. The
    equation is numerically integrated using the 4th order runge kutta scheme
    for an overdamped equation (ODE order 1). We appromate the derivative at t
    (1rst step rk4) with a forward time derivative and the derivative at 
    t+delta_t (4th step rk4) with a backward time derivative.

    dependencies:
    - up, vp, particle velocity (particle variables)
    - tau, stokes relaxation time particle (particle variable)
    - B, buoyancy particle (particle variable)
    - Omega_earth, angular velocity earth (fieldset constant)
    - delta_x, delta_y, delta_t step for finite difference method gradients
      (fieldset constants)
    """
    #VELOCITY DEPENDEND DRAG STOKES TIME
    #get velocity flow and particle in m/s
    (uf1, vf1) = fieldset.UV[time, particle.depth,
                             particle.lat, particle.lon]
    
    uslip = (particle.up - uf1)* fieldset.Rearth * math.cos(particle.lat * math.pi /180) *  math.pi / 180.
    vslip = (particle.vp - vf1)* fieldset.Rearth * math.pi / 180
    if(fieldset.save_slip_velocity == True):
        particle.uslip = uslip
        particle.vslip = vslip
    #calculate Reynolds number
    Rep = math.sqrt((uslip)**2 +(vslip)**2) * particle.diameter / (fieldset.nu)
    if(Rep > 5000):
        Rep = 5000
    #calulate correction factor
    f_REp = 1 + Rep / (4. * (1 + math.sqrt(Rep))) + Rep / 60.
  

    # Stokes time based on correction factor
    tau = ( (1. + 2. * particle.B) * particle.diameter**2) /(36 * fieldset.nu * f_REp)

    Bterm_tau = (2 * (1. - particle.B) / (1. + 2. * particle.B)) * tau

    norm_deltax = 1.0 / (2.0 * fieldset.delta_x)
    norm_deltay = 1.0 / (2.0 * fieldset.delta_y)
    norm_deltat = 1.0 / (1.0 * particle.dt)
    
    # RK4 STEP 1
    # fluid field velocity at location of particle


    
    if(fieldset.gradient == False):
        DuDt1 = 0
        DvDt1 = 0
    else:
        # calculate time derivative of fluid field
        (uf_tp1, vf_tp1) = fieldset.UV[time+particle.dt,
                                    particle.depth, particle.lat, particle.lon]
        (uf_tm1, vf_tm1) = fieldset.UV[time, particle.depth,
                                    particle.lat, particle.lon]
        dudt1 = (uf_tp1 - uf_tm1) * norm_deltat
        dvdt1 = (vf_tp1 - vf_tm1) * norm_deltat

        # calculate spatial gradients fluid field
        (u_dxm1, v_dxm1) = fieldset.UV[time, particle.depth,
                                    particle.lat, particle.lon-fieldset.delta_x]
        (u_dxp1, v_dxp1) = fieldset.UV[time, particle.depth,
                                    particle.lat, particle.lon+fieldset.delta_x]
        (u_dym1, v_dym1) = fieldset.UV[time, particle.depth,
                                    particle.lat-fieldset.delta_y, particle.lon]
        (u_dyp1, v_dyp1) = fieldset.UV[time, particle.depth,
                                    particle.lat+fieldset.delta_y, particle.lon]
        dudx1 = (u_dxp1 - u_dxm1) * norm_deltax
        dudy1 = (u_dyp1 - u_dym1) * norm_deltay
        dvdx1 = (v_dxp1 - v_dxm1) * norm_deltax
        dvdy1 = (v_dyp1 - v_dym1) * norm_deltay

        # caluclate material derivative fluid
        DuDt1 =   (dudt1 + uf1 * dudx1 + vf1 * dudy1 ) 
        DvDt1 =   (dvdt1 + uf1 * dvdx1 + vf1 * dvdy1 ) 

    # coriolis force
    f1 =  2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor1 = -vf1 * f1
    vcor1 = uf1 * f1

    u1 = uf1 + Bterm_tau * (DuDt1 + ucor1)
    v1 = vf1 + Bterm_tau * (DvDt1 + vcor1)
    
    # lon, lat for next step
    lon1 = particle.lon + 0.5 * u1 * particle.dt
    lat1 = particle.lat + 0.5 * v1 * particle.dt
    time1 = time + 0.5 * particle.dt
    # RK4 STEP 2
    # fluid field velocity at location of particle
    (uf2, vf2) = fieldset.UV[time1, particle.depth, lat1, lon1]
    if(fieldset.gradient ==False):
        DuDt2 = 0
        DvDt2 =0
    else:
        # calculate time derivative of fluid field
        (uf_tp2, vf_tp2) = fieldset.UV[time+particle.dt,
                                    particle.depth, lat1, lon1]
        (uf_tm2, vf_tm2) = fieldset.UV[time,
                                    particle.depth, lat1, lon1]
        dudt2 = (uf_tp2 - uf_tm2) * norm_deltat
        dvdt2 = (vf_tp2 - vf_tm2) * norm_deltat

        # calculate spatial gradients fluid field
        (u_dxm2, v_dxm2) = fieldset.UV[time1, particle.depth,
                                    lat1, lon1 - fieldset.delta_x]
        (u_dxp2, v_dxp2) = fieldset.UV[time1, particle.depth,
                                    lat1, lon1 + fieldset.delta_x]
        (u_dym2, v_dym2) = fieldset.UV[time1, particle.depth,
                                    lat1 - fieldset.delta_y, lon1]
        (u_dyp2, v_dyp2) = fieldset.UV[time1, particle.depth,
                                    lat1 + fieldset.delta_y, lon1]
        dudx2 = (u_dxp2 - u_dxm2) * norm_deltax
        dudy2 = (u_dyp2 - u_dym2) * norm_deltay
        dvdx2 = (v_dxp2 - v_dxm2) * norm_deltax
        dvdy2 = (v_dyp2 - v_dym2) * norm_deltay

        # caluclate material derivative fluid
        DuDt2 =   ( dudt2 + uf2 * dudx2 + vf2 * dudy2 ) 
        DvDt2 =   ( dvdt2 + uf2 * dvdx2 + vf2 * dvdy2 )
    # coriolis force
    f2 =  2 * fieldset.Omega_earth * math.sin(lat1 * math.pi / 180)
    ucor2 = -vf2 * f2
    vcor2 = uf2 * f2

    u2 = uf2 + Bterm_tau * (DuDt2 + ucor2)
    v2 = vf2 + Bterm_tau * (DvDt2 + vcor2)

    # lon, lat for next step
    lon2 = particle.lon + 0.5 * u2 * particle.dt
    lat2 = particle.lat + 0.5 * v2 * particle.dt
    time2 = time + 0.5 * particle.dt

    # RK4 STEP 3
    # fluid field velocity at location of particle
    (uf3, vf3) = fieldset.UV[time2, particle.depth, lat2, lon2]
    if(fieldset.gradient == False):
        DuDt3 = 0
        DvDt3 = 0
    else:
        # calculate time derivative of fluid field
        (uf_tp3, vf_tp3) = fieldset.UV[time+particle.dt,
                                    particle.depth, lat2, lon2]
        (uf_tm3, vf_tm3) = fieldset.UV[time,
                                    particle.depth, lat2, lon2]
        dudt3 = (uf_tp3 - uf_tm3) * norm_deltat
        dvdt3 = (vf_tp3 - vf_tm3) * norm_deltat

        # calculate spatial gradients fluid field
        (u_dxm3, v_dxm3) = fieldset.UV[time2, particle.depth,
                                    lat2, lon2 - fieldset.delta_x]
        (u_dxp3, v_dxp3) = fieldset.UV[time2, particle.depth,
                                    lat2, lon2 + fieldset.delta_x]
        (u_dym3, v_dym3) = fieldset.UV[time2, particle.depth,
                                    lat2 - fieldset.delta_y, lon2]
        (u_dyp3, v_dyp3) = fieldset.UV[time2, particle.depth,
                                    lat2 + fieldset.delta_y, lon2]
        dudx3 = (u_dxp3 - u_dxm3) * norm_deltax
        dudy3 = (u_dyp3 - u_dym3) * norm_deltay
        dvdx3 = (v_dxp3 - v_dxm3) * norm_deltax
        dvdy3 = (v_dyp3 - v_dym3) * norm_deltay

        # caluclate material derivative fluid
        DuDt3 =   (dudt3 + uf3 * dudx3 + vf3 * dudy3)
        DvDt3 =   (dvdt3 + uf3 * dvdx3 + vf3 * dvdy3)

    # coriolis force
    f3 = 2 * fieldset.Omega_earth * math.sin(lat2 * math.pi / 180)
    ucor3 = -vf3 * f3
    vcor3 = uf3 * f3

    u3 = uf3 + Bterm_tau * (DuDt3 + ucor3)
    v3 = vf3 + Bterm_tau * (DvDt3 + vcor3)

    # lon, lat for next step
    lon3 = particle.lon + u3 * particle.dt
    lat3 = particle.lat + v3 * particle.dt
    time3 = time + particle.dt

    # RK4 STEP 4
    # fluid field velocity at location of particle
    (uf4, vf4) = fieldset.UV[time3, particle.depth, lat3, lon3]
    if(fieldset.gradient == False):
        DuDt4 = 0 
        DvDt4 = 0 
    else:
        # calculate time derivative of fluid field
        (uf_tp4, vf_tp4) = fieldset.UV[time + particle.dt,
                                    particle.depth, lat3, lon3]
        (uf_tm4, vf_tm4) = fieldset.UV[time, particle.depth,
                                    lat3, lon3]
        dudt4 = (uf_tp4 - uf_tm4) * norm_deltat
        dvdt4 = (vf_tp4 - vf_tm4) * norm_deltat

        # calculate spatial gradients fluid field
        (u_dxm4, v_dxm4) = fieldset.UV[time, particle.depth,
                                    lat3, lon3 - fieldset.delta_x]
        (u_dxp4, v_dxp4) = fieldset.UV[time, particle.depth,
                                    lat3, lon3 + fieldset.delta_x]
        (u_dym4, v_dym4) = fieldset.UV[time, particle.depth,
                                    lat3 - fieldset.delta_y, lon3]
        (u_dyp4, v_dyp4) = fieldset.UV[time, particle.depth,
                                    lat3 + fieldset.delta_y, lon3]
        dudx4 = (u_dxp4 - u_dxm4) * norm_deltax
        dudy4 = (u_dyp4 - u_dym4) * norm_deltay
        dvdx4 = (v_dxp4 - v_dxm4) * norm_deltax
        dvdy4 = (v_dyp4 - v_dym4) * norm_deltay

        # caluclate material derivative fluid
        DuDt4 =( dudt4 + uf4 * dudx4 + vf4 * dudy4)
        DvDt4 = (dvdt4 + uf4 * dvdx4 + vf4 * dvdy4)

    # coriolis force
    f4 = 2 * fieldset.Omega_earth * math.sin(lat3 * math.pi / 180)
    ucor4 = -vf4 * f4
    vcor4 = uf4 * f4

    u4 = uf4 + Bterm_tau * (DuDt4 + ucor4)
    v4 = vf4 + Bterm_tau * (DvDt4 + vcor4)

    # RK4 INTEGRATION STEP
    particle_dlon += (u1 + 2 * u2 + 2 * u3 + u4) * particle.dt / 6.0
    particle_dlat += (v1 + 2 * v2 + 2 * v3 + v4) * particle.dt / 6.0
    particle.up = (u1+ 2 * u2 + 2 * u3 + u4) / 6.0
    particle.vp =  (v1 + 2 * v2 + 2 * v3 + v4) / 6.0

def MRSMAdvectionRK4_2D_drag_Rep_constant(particle, fieldset, time):
    """
    Advection of particles using the slow manifold approximation of the Maxey-
    Riley equation in 2D without Basset history term and Faxen corrections. The
    equation is numerically integrated using the 4th order runge kutta scheme
    for an overdamped equation (ODE order 1). We appromate the derivative at t
    (1rst step rk4) with a forward time derivative and the derivative at 
    t+delta_t (4th step rk4) with a backward time derivative.

    dependencies:
    - up, vp, particle velocity (particle variables)
    - tau, stokes relaxation time particle (particle variable)
    - B, buoyancy particle (particle variable)
    - Omega_earth, angular velocity earth (fieldset constant)
    - delta_x, delta_y, delta_t step for finite difference method gradients
      (fieldset constants)
    """
    #VELOCITY DEPENDEND DRAG STOKES TIME
    #get velocity flow and particle in m/s
    (uf1, vf1) = fieldset.UV[time, particle.depth,
                             particle.lat, particle.lon]
    
    uslip = (particle.up - uf1)* fieldset.Rearth * math.cos(particle.lat * math.pi /180) *  math.pi / 180.
    vslip = (particle.vp - vf1)* fieldset.Rearth * math.pi / 180
    if(fieldset.save_slip_velocity == True):
        particle.uslip = uslip
        particle.vslip = vslip
    


    # Stokes time based on correction factor
    tau = particle.tau / particle.C_Rep

    Bterm_tau = (2 * (1. - particle.B) / (1. + 2. * particle.B)) * tau

    norm_deltax = 1.0 / (2.0 * fieldset.delta_x)
    norm_deltay = 1.0 / (2.0 * fieldset.delta_y)
    norm_deltat = 1.0 / (1.0 * particle.dt)
    
    # RK4 STEP 1
    # fluid field velocity at location of particle
    (uf1, vf1) = fieldset.UV[time, particle.depth,
                             particle.lat, particle.lon]
    if(fieldset.gradient == False):
        DuDt1 = 0
        DvDt1 = 0
    else:
        # calculate time derivative of fluid field
        (uf_tp1, vf_tp1) = fieldset.UV[time+particle.dt,
                                    particle.depth, particle.lat, particle.lon]
        (uf_tm1, vf_tm1) = fieldset.UV[time, particle.depth,
                                    particle.lat, particle.lon]
        dudt1 = (uf_tp1 - uf_tm1) * norm_deltat
        dvdt1 = (vf_tp1 - vf_tm1) * norm_deltat

        # calculate spatial gradients fluid field
        (u_dxm1, v_dxm1) = fieldset.UV[time, particle.depth,
                                    particle.lat, particle.lon-fieldset.delta_x]
        (u_dxp1, v_dxp1) = fieldset.UV[time, particle.depth,
                                    particle.lat, particle.lon+fieldset.delta_x]
        (u_dym1, v_dym1) = fieldset.UV[time, particle.depth,
                                    particle.lat-fieldset.delta_y, particle.lon]
        (u_dyp1, v_dyp1) = fieldset.UV[time, particle.depth,
                                    particle.lat+fieldset.delta_y, particle.lon]
        dudx1 = (u_dxp1 - u_dxm1) * norm_deltax
        dudy1 = (u_dyp1 - u_dym1) * norm_deltay
        dvdx1 = (v_dxp1 - v_dxm1) * norm_deltax
        dvdy1 = (v_dyp1 - v_dym1) * norm_deltay

        # caluclate material derivative fluid
        DuDt1 = (dudt1 + uf1 * dudx1 + vf1 * dudy1)
        DvDt1 = (dvdt1 + uf1 * dvdx1 + vf1 * dvdy1)

    # coriolis force
    f1 =  2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor1 = -vf1 * f1
    vcor1 = uf1 * f1

    u1 = uf1 + Bterm_tau * (DuDt1 + ucor1)
    v1 = vf1 + Bterm_tau * (DvDt1 + vcor1)
    
    # lon, lat for next step
    lon1 = particle.lon + 0.5 * u1 * particle.dt
    lat1 = particle.lat + 0.5 * v1 * particle.dt
    time1 = time + 0.5 * particle.dt
    # RK4 STEP 2
    # fluid field velocity at location of particle
    (uf2, vf2) = fieldset.UV[time1, particle.depth, lat1, lon1]
    
    if(fieldset.gradient == False):
        DuDt2 = 0
        DvDt2 =0
    else:
        # calculate time derivative of fluid field
        (uf_tp2, vf_tp2) = fieldset.UV[time+particle.dt,
                                    particle.depth, lat1, lon1]
        (uf_tm2, vf_tm2) = fieldset.UV[time,
                                    particle.depth, lat1, lon1]
        dudt2 = (uf_tp2 - uf_tm2) * norm_deltat
        dvdt2 = (vf_tp2 - vf_tm2) * norm_deltat

        # calculate spatial gradients fluid field
        (u_dxm2, v_dxm2) = fieldset.UV[time1, particle.depth,
                                    lat1, lon1 - fieldset.delta_x]
        (u_dxp2, v_dxp2) = fieldset.UV[time1, particle.depth,
                                    lat1, lon1 + fieldset.delta_x]
        (u_dym2, v_dym2) = fieldset.UV[time1, particle.depth,
                                    lat1 - fieldset.delta_y, lon1]
        (u_dyp2, v_dyp2) = fieldset.UV[time1, particle.depth,
                                    lat1 + fieldset.delta_y, lon1]
        dudx2 = (u_dxp2 - u_dxm2) * norm_deltax
        dudy2 = (u_dyp2 - u_dym2) * norm_deltay
        dvdx2 = (v_dxp2 - v_dxm2) * norm_deltax
        dvdy2 = (v_dyp2 - v_dym2) * norm_deltay

        # caluclate material derivative fluid
        DuDt2 = (dudt2 + uf2 * dudx2 + vf2 * dudy2)
        DvDt2 = (dvdt2 + uf2 * dvdx2 + vf2 * dvdy2)

    # coriolis force
    f2 =  2 * fieldset.Omega_earth * math.sin(lat1 * math.pi / 180)
    ucor2 = -vf2 * f2
    vcor2 = uf2 * f2

    u2 = uf2 + Bterm_tau * (DuDt2 + ucor2)
    v2 = vf2 + Bterm_tau * (DvDt2 + vcor2)

    # lon, lat for next step
    lon2 = particle.lon + 0.5 * u2 * particle.dt
    lat2 = particle.lat + 0.5 * v2 * particle.dt
    time2 = time + 0.5 * particle.dt

    # RK4 STEP 3
    # fluid field velocity at location of particle
    (uf3, vf3) = fieldset.UV[time2, particle.depth, lat2, lon2]
    if(fieldset.gradient == False):
        DuDt3 = 0
        DvDt3 =0
    else:
        # calculate time derivative of fluid field
        (uf_tp3, vf_tp3) = fieldset.UV[time+particle.dt,
                                    particle.depth, lat2, lon2]
        (uf_tm3, vf_tm3) = fieldset.UV[time,
                                    particle.depth, lat2, lon2]
        dudt3 = (uf_tp3 - uf_tm3) * norm_deltat
        dvdt3 = (vf_tp3 - vf_tm3) * norm_deltat

        # calculate spatial gradients fluid field
        (u_dxm3, v_dxm3) = fieldset.UV[time2, particle.depth,
                                    lat2, lon2 - fieldset.delta_x]
        (u_dxp3, v_dxp3) = fieldset.UV[time2, particle.depth,
                                    lat2, lon2 + fieldset.delta_x]
        (u_dym3, v_dym3) = fieldset.UV[time2, particle.depth,
                                    lat2 - fieldset.delta_y, lon2]
        (u_dyp3, v_dyp3) = fieldset.UV[time2, particle.depth,
                                    lat2 + fieldset.delta_y, lon2]
        dudx3 = (u_dxp3 - u_dxm3) * norm_deltax
        dudy3 = (u_dyp3 - u_dym3) * norm_deltay
        dvdx3 = (v_dxp3 - v_dxm3) * norm_deltax
        dvdy3 = (v_dyp3 - v_dym3) * norm_deltay

        # caluclate material derivative fluid
        DuDt3 = dudt3 + uf3 * dudx3 + vf3 * dudy3
        DvDt3 = dvdt3 + uf3 * dvdx3 + vf3 * dvdy3

    # coriolis force
    f3 = 2 * fieldset.Omega_earth * math.sin(lat2 * math.pi / 180)
    ucor3 = -vf3 * f3
    vcor3 = uf3 * f3

    u3 = uf3 + Bterm_tau * (DuDt3 + ucor3)
    v3 = vf3 + Bterm_tau * (DvDt3 + vcor3)

    # lon, lat for next step
    lon3 = particle.lon + u3 * particle.dt
    lat3 = particle.lat + v3 * particle.dt
    time3 = time + particle.dt

    # RK4 STEP 4
    # fluid field velocity at location of particle
    (uf4, vf4) = fieldset.UV[time3, particle.depth, lat3, lon3]
    if(fieldset.gradient == False):
        DuDt4 = 0
        DvDt4 =0
    else:
        # calculate time derivative of fluid field
        (uf_tp4, vf_tp4) = fieldset.UV[time + particle.dt,
                                    particle.depth, lat3, lon3]
        (uf_tm4, vf_tm4) = fieldset.UV[time, particle.depth,
                                    lat3, lon3]
        dudt4 = (uf_tp4 - uf_tm4) * norm_deltat
        dvdt4 = (vf_tp4 - vf_tm4) * norm_deltat

        # calculate spatial gradients fluid field
        (u_dxm4, v_dxm4) = fieldset.UV[time, particle.depth,
                                    lat3, lon3 - fieldset.delta_x]
        (u_dxp4, v_dxp4) = fieldset.UV[time, particle.depth,
                                    lat3, lon3 + fieldset.delta_x]
        (u_dym4, v_dym4) = fieldset.UV[time, particle.depth,
                                    lat3 - fieldset.delta_y, lon3]
        (u_dyp4, v_dyp4) = fieldset.UV[time, particle.depth,
                                    lat3 + fieldset.delta_y, lon3]
        dudx4 = (u_dxp4 - u_dxm4) * norm_deltax
        dudy4 = (u_dyp4 - u_dym4) * norm_deltay
        dvdx4 = (v_dxp4 - v_dxm4) * norm_deltax
        dvdy4 = (v_dyp4 - v_dym4) * norm_deltay

        # caluclate material derivative fluid
        DuDt4 = dudt4 + uf4 * dudx4 + vf4 * dudy4
        DvDt4 = dvdt4 + uf4 * dvdx4 + vf4 * dvdy4

    # coriolis force
    f4 = 2 * fieldset.Omega_earth * math.sin(lat3 * math.pi / 180)
    ucor4 = -vf4 * f4
    vcor4 = uf4 * f4

    u4 = uf4 + Bterm_tau * (DuDt4 + ucor4)
    v4 = vf4 + Bterm_tau * (DvDt4 + vcor4)

    # RK4 INTEGRATION STEP
    particle_dlon += (u1 + 2 * u2 + 2 * u3 + u4) * particle.dt / 6.0
    particle_dlat += (v1 + 2 * v2 + 2 * v3 + v4) * particle.dt / 6.0
    particle.up = (u1+ 2 * u2 + 2 * u3 + u4) / 6.0
    particle.vp =  (v1 + 2 * v2 + 2 * v3 + v4) / 6.0


