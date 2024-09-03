from parcels import JITParticle, Variable
import math
import numpy as np


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
    wp = Variable('wp', dtype=np.float32, to_write=False, initial=0.)


def deleteParticle(particle, fieldset, time):
    """ Kernel for deleting particles if they throw an error other than through
    the surface
    """
    if particle.state >= 50:
        particle.delete()


def InitializeParticles2D(particle, fieldset, time):
    """Kernel for initializing the velocity of the intertial particles in 2D
    to the value of the fluid at the location of the particle.
    """
    u, v = fieldset.UV[particle]
    particle.up = u
    particle.vp = v


def InitializeParticles3D(particle, fieldset, time):
    """Kernel for initializing the velocity of the intertial particles in 3D
    to the value of the fluid at the location of the particle for x/y (lon/lat)
    in the z direction the velocity is given by the fluid velocity + the stokes
    settling velocity (w0).
    """
    u, v, w = fieldset.UVW[particle]
    w0 = (2 * (1 - particle.B) / (1 + 2 * particle.B)
          * fieldset.g * particle.tau)
    particle.up = u
    particle.vp = v
    particle.wp = w + w0


def MRAdvectionEC_2D(particle, fieldset, time):
    """
    Advection of particles using Maxey-Riley equation in 2D without Basset
    history term and Faxen corrections without sinking or floating force. The
    equation is numerically integrated using the Euler-Cromes scheme

    dependencies:
    the particle should contain variables up, vp, the velocity of the particle,
    tau, the stokes time, and B, the buoyancy of the particle.
    The fieldset should contain constants Omega_earth (angular velocity earth),
    delta_x, step for finite difference method gradients)
    """
    tau_inv = 1. / particle.tau
    Bterm = 3. / (1. + 2. * particle.B)
    # read in velocity at location of particle
    (uf, vf) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    # calculate time derivative of fluid field
    (uf_tp, vf_tp) = fieldset.UV[time+particle.dt,
                                 particle.depth, particle.lat, particle.lon]
    norm_deltat = 1. / particle.dt
    dudt = (uf_tp - uf) * norm_deltat
    dvdt = (vf_tp - vf) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm, v_dxm) = fieldset.UV[time, particle.depth,
                                 particle.lat, particle.lon - fieldset.delta_x]
    (u_dxp, v_dxp) = fieldset.UV[time, particle.depth,
                                 particle.lat, particle.lon + fieldset.delta_x]
    (u_dym, v_dym) = fieldset.UV[time, particle.depth,
                                 particle.lat - fieldset.delta_y, particle.lon]
    (u_dyp, v_dyp) = fieldset.UV[time, particle.depth,
                                 particle.lat + fieldset.delta_y, particle.lon]
    norm_deltax = 1.0 / (2.0 * fieldset.delta_x)
    norm_deltay = 1.0 / (2.0 * fieldset.delta_y)
    dudx = (u_dxp - u_dxm) * norm_deltax
    dudy = (u_dyp - u_dym) * norm_deltay
    dvdx = (v_dxp - v_dxm) * norm_deltax
    dvdy = (v_dyp - v_dym) * norm_deltay

    # caluclate material derivative fluid
    DuDt = dudt + uf * dudx + vf * dudy
    DvDt = dvdt + uf * dvdx + vf * dvdy

    # coriolis force
    f = 0  # 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi/180)
    ucor = -vf * f
    vcor = uf * f

    # drag force
    udrag = tau_inv * (uf - particle.up)
    vdrag = tau_inv * (vf - particle.vp)

    # advection using the Euler-Cromer algorithm:
    a_lon = Bterm * (DuDt + ucor) + udrag
    a_lat = Bterm * (DvDt + vcor) + vdrag

    particle.up = particle.up + a_lon * particle.dt
    particle.vp = particle.vp + a_lat * particle.dt

    particle_dlon += particle.up * particle.dt
    particle_dlat += particle.vp * particle.dt


def MRAdvectionRK4_2D(particle, fieldset, time):
    """
    Advection of particles using Maxey-Riley equation in 2D without Basset
    history term and Faxen corrections without sinking or floating force.
    The equation is numerically integrated using the 4th order runge kutta
    scheme for a 2nd order ODE equation.

    dependencies:
    the particle should contain variables up, vp, the velocity of the particle,
    tau, the stokes time, and B, the buoyancy of the particle.
    The fieldset should contain Omega_earth, the angular velocity of the earth,
    and delta_x, delta_y, step for finite difference method gradients. We
    appromate the time derivative at t (1rst step rk4) with a forward finite
    difference and the time derivative at t+delta_t (4th step rk4) with a
    backward finite difference.
    """
    tau_inv = 1. / particle.tau
    Bterm = 3. / (1. + 2. * particle.B)

    norm_deltax = 1.0 / (2.0 * fieldset.delta_x)
    norm_deltay = 1.0 / (2.0 * fieldset.delta_y)
    norm_deltat = 1.0 / (1.0 * particle.dt)

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
    (uf_tm1, vf_tm1) = fieldset.UV[time+particle.dt,
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

    # coriolis force
    f1 = 0  # 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor1 = -vf1 * f1
    vcor1 = uf1 * f1

    # drag force
    udrag1 = tau_inv * (uf1 - up1)
    vdrag1 = tau_inv * (vf1 - vp1)

    # acceleration
    a_lon1 = Bterm * (DuDt1 + ucor1) + udrag1
    a_lat1 = Bterm * (DvDt1 + vcor1) + vdrag1

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

    # coriolis force
    f2 = 0  # 2 * fieldset.Omega_earth * math.sin(lat1 * math.pi / 180.)
    ucor2 = -vf2 * f2
    vcor2 = uf2 * f2

    # drag force
    udrag2 = tau_inv * (uf2 - up2)
    vdrag2 = tau_inv * (vf2 - vp2)

    # acceleration
    a_lon2 = Bterm * (DuDt2 + ucor2) + udrag2
    a_lat2 = Bterm * (DvDt2 + vcor2) + vdrag2

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

    # coriolis force
    f3 = 0  # 2 * fieldset.Omega_earth * math.sin(lat2 * math.pi / 180.)
    ucor3 = -vf3 * f3
    vcor3 = uf3 * f3

    # drag force
    udrag3 = tau_inv * (uf3 - up3)
    vdrag3 = tau_inv * (vf3 - vp3)

    # acceleration
    a_lon3 = Bterm * (DuDt3 + ucor3) + udrag3
    a_lat3 = Bterm * (DvDt3 + vcor3) + vdrag3

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

    # coriolis force
    f4 = 0  # 2 * fieldset.Omega_earth * math.sin(lat3 * math.pi / 180)
    ucor4 = -vf4 * f4
    vcor4 = uf4 * f4

    # drag force
    udrag4 = tau_inv * (uf4-up4)
    vdrag4 = tau_inv * (vf4-vp4)

    # acceleration
    a_lon4 = Bterm * (DuDt4 + ucor4) + udrag4
    a_lat4 = Bterm * (DvDt4 + vcor4) + vdrag4

    # RK4 INTEGRATION STEP
    particle.up += (a_lon1 + 2 * a_lon2
                    + 2 * a_lon3 + a_lon4) * particle.dt / 6.0
    particle.vp += (a_lat1 + 2 * a_lat2
                    + 2 * a_lat3 + a_lat4) * particle.dt / 6.0
    particle_dlon += (up1 + 2 * up2 + 2 * up3 + up4) * particle.dt / 6.0
    particle_dlat += (vp1 + 2 * vp2 + 2 * vp3 + vp4) * particle.dt / 6.0


def MRAdvectionEC_3D(particle, fieldset, time):
    """
    Advection of particles using Maxey-Riley equation in 3D without Basset
    history term and Faxen corrections. The equation is numerically
    integrated using the Euler-Cromes scheme

    dependencies:
    the particle should contain variables up, vp, wp, the velocity of the
    particle, tau, the stokes time, and B, the buoyancy of the particle.
    The fieldset should contain Omega_earth, the angular velocity of the earth,
    and delta_x, delta_y, delta_z, step for finite difference method gradients.
    """
    tau_inv = 1. / particle.tau
    Bterm = (3. / (1. + 2. * particle.B))
    w0overtau = 2 * (1 - particle.B) / (1 + 2 * particle.B) * fieldset.g
    # read in velocity at location of particle
    (uf, vf, wf) = fieldset.UVW[time, particle.depth,
                                particle.lat, particle.lon]

    # calculate time derivative of fluid field
    (uf_tp, vf_tp, wf_tp) = fieldset.UVW[time+particle.dt, particle.depth,
                                         particle.lat, particle.lon]
    norm_deltat = 1. / (2.0 * fieldset.delta_t)
    dudt = (uf_tp - uf) * norm_deltat
    dvdt = (vf_tp - vf) * norm_deltat
    dwdt = (wf_tp - wf) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm, v_dxm, w_dxm) = fieldset.UVW[time, particle.depth, particle.lat,
                                         particle.lon-fieldset.delta_x]
    (u_dxp, v_dxp, w_dxp) = fieldset.UVW[time, particle.depth, particle.lat,
                                         particle.lon+fieldset.delta_x]
    (u_dym, v_dym, w_dym) = fieldset.UVW[time, particle.depth, particle.lat
                                         - fieldset.delta_y, particle.lon]
    (u_dyp, v_dyp, w_dyp) = fieldset.UVW[time, particle.depth, particle.lat
                                         + fieldset.delta_y, particle.lon]
    (u_dzm, v_dzm, w_dzm) = fieldset.UVW[time, particle.depth
                                         - fieldset.delta_z,
                                         particle.lat, particle.lon]
    (u_dzp, v_dzp, w_dzp) = fieldset.UVW[time, particle.depth
                                         + fieldset.delta_z,
                                         particle.lat, particle.lon]
    norm_deltax = 1.0 / (2.0 * fieldset.delta_x)
    norm_deltay = 1.0 / (2.0 * fieldset.delta_y)
    norm_deltaz = 1.0 / (2.0 * fieldset.delta_z)
    dudx = (u_dxp - u_dxm) * norm_deltax
    dudy = (u_dyp - u_dym) * norm_deltay
    dudz = (u_dzp - u_dzm) * norm_deltaz
    dvdx = (v_dxp - v_dxm) * norm_deltax
    dvdy = (v_dyp - v_dym) * norm_deltay
    dvdz = (v_dzp - v_dzm) * norm_deltaz
    dwdx = (w_dxp - w_dxm) * norm_deltax
    dwdy = (w_dyp - w_dym) * norm_deltay
    dwdz = (w_dzp - w_dzm) * norm_deltaz

    # caluclate material derivative fluid
    DuDt = dudt + uf * dudx + vf * dudy + wf * dudz
    DvDt = dvdt + uf * dvdx + vf * dvdy + wf * dvdz
    DwDt = dwdt + uf * dwdx + vf * dwdy + wf * dwdz

    # coriolis force
    f = 0  # 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor = -vf * f
    vcor = uf * f

    # drag force
    udrag = tau_inv * (uf - particle.up)
    vdrag = tau_inv * (vf - particle.vp)
    wdrag = tau_inv * (wf - particle.wp)

    # advection using the Euler-Cromer algorithm:
    a_lon = Bterm * (DuDt + ucor) + udrag
    a_lat = Bterm * (DvDt + vcor) + vdrag
    a_depth = Bterm * (DwDt) + wdrag + w0overtau

    particle.up = particle.up + a_lon * particle.dt
    particle.vp = particle.vp + a_lat * particle.dt
    particle.wp = particle.wp + a_depth * particle.dt

    particle_dlon += particle.up * particle.dt
    particle_dlat += particle.vp * particle.dt
    particle_ddepth += particle.wp * particle.dt


def MRAdvectionRK4_3D(particle, fieldset, time):
    """
    Advection of particles using Maxey-Riley equation in 3D
    without Basset history term and Faxen corrections
    The equation is numerically integrated using the 4th
    order runge kutta scheme for a nonoverdamped equation.

    dependencies:
    the particle should contain up, vp, wp (velocity particle
    at timestep t),tau_inv is 1/stokes time,
    Bterm=3/(1+2B) with B the buoyancy of the particle and
    w0 is the stokes terminal velocity devided by tau
    The fieldset should contain Omega_earth, the angular velocity of the earth,
    and delta_x, delta_y, delta_z, step for finite difference method gradients.
    We appromate the derivative at t (1rst step rk4) with a forward time
    derivative and the derivative at t+delta_t (4th step rk4) with a backward
    time derivative.
    """
    tau_inv = 1. / particle.tau
    Bterm = (3. / (1. + 2. * particle.B))
    w0 = (2 * (1 - particle.B) / (1 + 2 * particle.B)
          * fieldset.g * particle.tau)
    norm_deltax = 1.0 / (2.0 * fieldset.delta_x)
    norm_deltay = 1.0 / (2.0 * fieldset.delta_y)
    norm_deltaz = 1.0 / (2.0 * fieldset.delta_z)
    norm_deltat = 1.0 / (1.0 * particle.dt)

    # RK4 STEP 1
    # velocity particle at current step
    up1 = particle.up
    vp1 = particle.vp
    wp1 = particle.wp

    # read in velocity field at location of particle
    (uf1, vf1, wf1) = fieldset.UVW[time, particle.depth,
                                   particle.lat, particle.lon]

    # calculate time derivative of fluid field
    (uf_tp1, vf_tp1, wf_tp1) = fieldset.UVW[time + particle.dt, particle.depth,
                                            particle.lat, particle.lon]
    (uf_tm1, vf_tm1, wf_tm1) = fieldset.UVW[time, particle.depth,
                                            particle.lat, particle.lon]
    dudt1 = (uf_tp1 - uf_tm1) * norm_deltat
    dvdt1 = (vf_tp1 - vf_tm1) * norm_deltat
    dwdt1 = (wf_tp1 - wf_tm1) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm1, v_dxm1, w_dxm1) = fieldset.UVW[time, particle.depth,
                                            particle.lat, particle.lon
                                            - fieldset.delta_x]
    (u_dxp1, v_dxp1, w_dxp1) = fieldset.UVW[time, particle.depth,
                                            particle.lat, particle.lon
                                            + fieldset.delta_x]
    (u_dym1, v_dym1, w_dym1) = fieldset.UVW[time, particle.depth,
                                            particle.lat - fieldset.delta_y,
                                            particle.lon]
    (u_dyp1, v_dyp1, w_dyp1) = fieldset.UVW[time, particle.depth,
                                            particle.lat + fieldset.delta_y,
                                            particle.lon]
    (u_dzm1, v_dzm1, w_dzm1) = fieldset.UVW[time, particle.depth
                                            - fieldset.delta_z, particle.lat,
                                            particle.lon]
    (u_dzp1, v_dzp1, w_dzp1) = fieldset.UVW[time, particle.depth
                                            + fieldset.delta_z, particle.lat,
                                            particle.lon]
    dudx1 = (u_dxp1 - u_dxm1) * norm_deltax
    dudy1 = (u_dyp1 - u_dym1) * norm_deltay
    dudz1 = (u_dzp1 - u_dzm1) * norm_deltaz
    dvdx1 = (v_dxp1 - v_dxm1) * norm_deltax
    dvdy1 = (v_dyp1 - v_dym1) * norm_deltay
    dvdz1 = (v_dzp1 - v_dzm1) * norm_deltaz
    dwdx1 = (w_dxp1 - w_dxm1) * norm_deltax
    dwdy1 = (w_dyp1 - w_dym1) * norm_deltay
    dwdz1 = (w_dzp1 - w_dzm1) * norm_deltaz

    # caluclate material derivative fluid
    DuDt1 = dudt1 + uf1 * dudx1 + vf1 * dudy1 + wf1 * dudz1
    DvDt1 = dvdt1 + uf1 * dvdx1 + vf1 * dvdy1 + wf1 * dvdz1
    DwDt1 = dwdt1 + uf1 * dwdx1 + vf1 * dwdy1 + wf1 * dwdz1

    # coriolis force
    f1 = 0  # 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor1 = -vf1 * f1
    vcor1 = uf1 * f1

    # drag force
    udrag1 = tau_inv * (uf1 - up1)
    vdrag1 = tau_inv * (vf1 - vp1)
    wdrag1 = tau_inv * (w0 + wf1 - wp1)

    # acceleration particle for current step
    a_lon1 = Bterm * (DuDt1 + ucor1) + udrag1
    a_lat1 = Bterm * (DvDt1 + vcor1) + vdrag1
    a_depth1 = Bterm * DwDt1 + wdrag1

    # lon, lat, depth for next step
    lon1 = particle.lon + 0.5 * up1 * particle.dt
    lat1 = particle.lat + 0.5 * vp1 * particle.dt
    depth1 = particle.depth + 0.5 * wp1 * particle.dt
    time1 = time + 0.5 * particle.dt

    # RK4 STEP 2
    # velocity particle at current step
    up2 = particle.up + 0.5 * a_lon1 * particle.dt
    vp2 = particle.vp + 0.5 * a_lat1 * particle.dt
    wp2 = particle.wp + 0.5 * a_depth1 * particle.dt

    # read in velocity at location of particle
    (uf2, vf2, wf2) = fieldset.UVW[time1, depth1, lat1, lon1]

    # calculate time derivative of fluid field
    (uf_tp2, vf_tp2, wf_tp2) = fieldset.UVW[time + particle.dt,
                                            depth1, lat1, lon1]
    (uf_tm2, vf_tm2, wf_tm2) = fieldset.UVW[time, depth1, lat1, lon1]
    dudt2 = (uf_tp2 - uf_tm2) * norm_deltat
    dvdt2 = (vf_tp2 - vf_tm2) * norm_deltat
    dwdt2 = (wf_tp2 - wf_tm2) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm2, v_dxm2, w_dxm2) = fieldset.UVW[time1, depth1,
                                            lat1, lon1 - fieldset.delta_x]
    (u_dxp2, v_dxp2, w_dxp2) = fieldset.UVW[time1, depth1,
                                            lat1, lon1 + fieldset.delta_x]
    (u_dym2, v_dym2, w_dym2) = fieldset.UVW[time1, depth1,
                                            lat1 - fieldset.delta_y, lon1]
    (u_dyp2, v_dyp2, w_dyp2) = fieldset.UVW[time1, depth1,
                                            lat1 + fieldset.delta_y, lon1]
    (u_dzm2, v_dzm2, w_dzm2) = fieldset.UVW[time1, depth1
                                            - fieldset.delta_z, lat1, lon1]
    (u_dzp2, v_dzp2, w_dzp2) = fieldset.UVW[time1, depth1
                                            + fieldset.delta_z, lat1, lon1]
    dudx2 = (u_dxp2 - u_dxm2) * norm_deltax
    dudy2 = (u_dyp2 - u_dym2) * norm_deltay
    dudz2 = (u_dzp2 - u_dzm2) * norm_deltaz
    dvdx2 = (v_dxp2 - v_dxm2) * norm_deltax
    dvdy2 = (v_dyp2 - v_dym2) * norm_deltay
    dvdz2 = (v_dzp2 - v_dzm2) * norm_deltaz
    dwdx2 = (w_dxp2 - w_dxm2) * norm_deltax
    dwdy2 = (w_dyp2 - w_dym2) * norm_deltay
    dwdz2 = (w_dzp2 - w_dzm2) * norm_deltaz

    # caluclate material derivative fluid
    DuDt2 = dudt2 + uf2 * dudx2 + vf2 * dudy2 + wf2 * dudz2
    DvDt2 = dvdt2 + uf2 * dvdx2 + vf2 * dvdy2 + wf2 * dvdz2
    DwDt2 = dwdt2 + uf2 * dwdx2 + vf2 * dwdy2 + wf2 * dwdz2

    # coriolis force
    f2 = 0  # 2 * fieldset.Omega_earth * math.sin(lat1 * math.pi / 180)
    ucor2 = -vf2 * f2
    vcor2 = uf2 * f2

    # drag force
    udrag2 = tau_inv * (uf2 - up2)
    vdrag2 = tau_inv * (vf2 - vp2)
    wdrag2 = tau_inv * (w0 + wf2 - wp2)

    # acceleration particle for current step
    a_lon2 = Bterm * (DuDt2 + ucor2) + udrag2
    a_lat2 = Bterm * (DvDt2+vcor2)+vdrag2
    a_depth2 = Bterm * DwDt2+wdrag2

    # calculate RK4 coefficients
    lon2 = particle.lon + 0.5 * up2 * particle.dt
    lat2 = particle.lat + 0.5 * vp2 * particle.dt
    depth2 = particle.depth + 0.5 * wp2 * particle.dt
    time2 = time + 0.5 * particle.dt

    # RK4 STEP 3
    # velocity particle at current step
    up3 = particle.up + 0.5 * a_lon2 * particle.dt
    vp3 = particle.vp + 0.5 * a_lat2 * particle.dt
    wp3 = particle.wp + 0.5 * a_depth2 * particle.dt

    (uf3, vf3, wf3) = fieldset.UVW[time2, depth2, lat2, lon2]

    # calculate time derivative of fluid field
    uf_tp3, vf_tp3, wf_tp3 = fieldset.UVW[time + particle.dt,
                                          depth2, lat2, lon2]
    uf_tm3, vf_tm3, wf_tm3 = fieldset.UVW[time, depth2, lat2, lon2]
    dudt3 = (uf_tp3 - uf_tm3) * norm_deltat
    dvdt3 = (vf_tp3 - vf_tm3) * norm_deltat
    dwdt3 = (wf_tp3 - wf_tm3) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm3, v_dxm3, w_dxm3) = fieldset.UVW[time2, depth2,
                                            lat2, lon2 - fieldset.delta_x]
    (u_dxp3, v_dxp3, w_dxp3) = fieldset.UVW[time2, depth2,
                                            lat2, lon2 + fieldset.delta_x]
    (u_dym3, v_dym3, w_dym3) = fieldset.UVW[time2, depth2,
                                            lat2 - fieldset.delta_y, lon2]
    (u_dyp3, v_dyp3, w_dyp3) = fieldset.UVW[time2, depth2,
                                            lat2+fieldset.delta_y, lon2]
    (u_dzm3, v_dzm3, w_dzm3) = fieldset.UVW[time2, depth2
                                            - fieldset.delta_z, lat2, lon2]
    (u_dzp3, v_dzp3, w_dzp3) = fieldset.UVW[time2, depth2
                                            + fieldset.delta_z, lat2, lon2]
    dudx3 = (u_dxp3 - u_dxm3) * norm_deltax
    dudy3 = (u_dyp3 - u_dym3) * norm_deltay
    dudz3 = (u_dzp3 - u_dzm3) * norm_deltaz
    dvdx3 = (v_dxp3 - v_dxm3) * norm_deltax
    dvdy3 = (v_dyp3 - v_dym3) * norm_deltay
    dvdz3 = (v_dzp3 - v_dzm3) * norm_deltaz
    dwdx3 = (w_dxp3 - w_dxm3) * norm_deltax
    dwdy3 = (w_dyp3 - w_dym3) * norm_deltay
    dwdz3 = (w_dzp3 - w_dzm3) * norm_deltaz

    # caluclate material derivative fluid
    DuDt3 = dudt3 + uf3 * dudx3 + vf3 * dudy3 + wf3 * dudz3
    DvDt3 = dvdt3 + uf3 * dvdx3 + vf3 * dvdy3 + wf3 * dvdz3
    DwDt3 = dwdt3 + uf3 * dwdx3 + vf3 * dwdy3 + wf3 * dwdz3

    # coriolis force
    f3 = 0  # 2 * fieldset.Omega_earth * math.sin(lat2 * math.pi / 180)
    ucor3 = -vf3 * f3
    vcor3 = uf3 * f3

    # drag force
    udrag3 = tau_inv * (uf3 - up3)
    vdrag3 = tau_inv * (vf3 - vp3)
    wdrag3 = tau_inv * (w0 + wf3 - wp3)

    # acceleration particle for current step
    a_lon3 = Bterm * (DuDt3 + ucor3) + udrag3
    a_lat3 = Bterm * (DvDt3 + vcor3) + vdrag3
    a_depth3 = Bterm * (DwDt3) + wdrag3

    # calculate RK4 coefficients
    lon3 = particle.lon + up3 * particle.dt
    lat3 = particle.lat + vp3 * particle.dt
    depth3 = particle.depth + wp3 * particle.dt
    time3 = time + particle.dt

    # RK4 STEP 4
    # velocity particle at current step
    up4 = particle.up + a_lon3 * particle.dt
    vp4 = particle.vp + a_lat3 * particle.dt
    wp4 = particle.wp + a_depth3 * particle.dt

    (uf4, vf4, wf4) = fieldset.UVW[time3, depth3, lat3, lon3]

    # calculate time derivative of fluid field
    (uf_tp4, vf_tp4, wf_tp4) = fieldset.UVW[time+particle.dt,
                                            depth3, lat3, lon3]
    (uf_tm4, vf_tm4, wf_tm4) = fieldset.UVW[time,  depth3, lat3, lon3]
    dudt4 = (uf_tp4 - uf_tm4) * norm_deltat
    dvdt4 = (vf_tp4 - vf_tm4) * norm_deltat
    dwdt4 = (wf_tp4 - wf_tm4) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm4, v_dxm4, w_dxm4) = fieldset.UVW[time3, depth3,
                                            lat3, lon3 - fieldset.delta_x]
    (u_dxp4, v_dxp4, w_dxp4) = fieldset.UVW[time3, depth3,
                                            lat3, lon3 + fieldset.delta_x]
    (u_dym4, v_dym4, w_dym4) = fieldset.UVW[time3, depth3,
                                            lat3 - fieldset.delta_y, lon3]
    (u_dyp4, v_dyp4, w_dyp4) = fieldset.UVW[time3, depth3,
                                            lat3 + fieldset.delta_y, lon3]
    (u_dzm4, v_dzm4, w_dzm4) = fieldset.UVW[time3, depth3
                                            - fieldset.delta_z, lat3, lon3]
    (u_dzp4, v_dzp4, w_dzp4) = fieldset.UVW[time3, depth3
                                            + fieldset.delta_z, lat3, lon3]
    dudx4 = (u_dxp4 - u_dxm4) * norm_deltax
    dudy4 = (u_dyp4 - u_dym4) * norm_deltay
    dudz4 = (u_dzp4 - u_dzm4) * norm_deltaz
    dvdx4 = (v_dxp4 - v_dxm4) * norm_deltax
    dvdy4 = (v_dyp4 - v_dym4) * norm_deltay
    dvdz4 = (v_dzp4 - v_dzm4) * norm_deltaz
    dwdx4 = (w_dxp4 - w_dxm4) * norm_deltax
    dwdy4 = (w_dyp4 - w_dym4) * norm_deltay
    dwdz4 = (w_dzp4 - w_dzm4) * norm_deltaz

    # caluclate material derivative fluid
    DuDt4 = dudt4 + uf4 * dudx4 + vf4 * dudy4 + wf4 * dudz4
    DvDt4 = dvdt4 + uf4 * dvdx4 + vf4 * dvdy4 + wf4 * dvdz4
    DwDt4 = dwdt4 + uf4 * dwdx4 + vf4 * dwdy4 + wf4 * dwdz4

    # coriolis force
    f4 = 0  # 2 * fieldset.Omega_earth * math.sin(lat3 * math.pi / 180)
    ucor4 = -vf4 * f4
    vcor4 = uf4 * f4

    # drag force
    udrag4 = tau_inv * (uf4 - up4)
    vdrag4 = tau_inv * (vf4 - vp4)
    wdrag4 = tau_inv * (w0 + wf4 - wp4)

    # acceleration particle for current step
    a_lon4 = Bterm * (DuDt4 + ucor4) + udrag4
    a_lat4 = Bterm * (DvDt4 + vcor4) + vdrag4
    a_depth4 = Bterm * DwDt4 + wdrag4

    # RK4 INTERGRATION STEP
    particle.up += (a_lon1 + 2 * a_lon2
                    + 2 * a_lon3 + a_lon4) * particle.dt / 6.
    particle.vp += (a_lat1 + 2 * a_lat2
                    + 2 * a_lat3 + a_lat4) * particle.dt / 6.
    particle.wp += (a_depth1 + 2 * a_depth2
                    + 2 * a_depth3 + a_depth4) * particle.dt / 6.
    particle_dlon += (up1 + 2 * up2 + 2 * up3 + up4) * particle.dt / 6.
    particle_dlat += (vp1 + 2 * vp2 + 2 * vp3 + vp4) * particle.dt / 6.
    particle_ddepth += (wp1 + 2 * wp2 + 2 * wp3 + wp4) * particle.dt / 6.


#  SLOW - MANIFOLD
def MRSMAdvectionRK4_2D(particle, fieldset, time):
    """
    Advection of particles using the slow manifold approximation
    of the Maxey-Riley equation in 2D without Basset history term
    and Faxen corrections. The equation is numerically integrated
    using the 4th order runge kutta scheme for an overdamped equation.
    (ODE order 1)

    dependencies:
    the particle should contain, tau, the stokes time, B, the buoyancy of the
    particle. The fieldset should contain Omega_earth, the angular velocity
    of the earth, and delta_x, delta_y, the step for finite
    difference method gradients. We appromate the time derivative at t
    (1rst step rk4) with a forward finite difference and the time derivative
    at t+delta_t (4th step rk4) with a backward finite difference.
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
    f1 = 0  # 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
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
    DuDt2 = dudt2 + uf2 * dudx2 + vf2 * dudy2
    DvDt2 = dvdt2 + uf2 * dvdx2 + vf2 * dvdy2

    # coriolis force
    f2 = 0  # 2 * fieldset.Omega_earth*math.sin(lat1 * math.pi / 180)
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
    f3 = 0  # 2 * fieldset.Omega_earth*math.sin(lat2 * math.pi / 180)
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
    f4 = 0  # 2 * fieldset.Omega_earth * math.sin(lat3 * math.pi / 180)
    ucor4 = -vf4 * f4
    vcor4 = uf4 * f4

    u4 = uf4 + Bterm_tau * (DuDt4 + ucor4)
    v4 = vf4 + Bterm_tau * (DvDt4 + vcor4)

    # RK4 INTEGRATION STEP
    particle_dlon += (u1 + 2 * u2 + 2 * u3 + u4) * particle.dt / 6.0
    particle_dlat += (v1 + 2 * v2 + 2 * v3 + v4) * particle.dt / 6.0


def MRSMAdvectionRK4_3D(particle, fieldset, time):
    """
    Advection of particles using the slow manifold approximation
    of the Maxey-Riley equation in 3D without Basset history term
    and Faxen corrections. The equation is numerically integrated
    using the 4th order runge kutta scheme for an overdamped equation.
    (ODE order 1)

    dependencies:
    the particle should contain, tau, the stokes time, B, the buoyancy of the
    particle. The fieldset should contain Omega_earth, the angular velocity
    of the earth, and delta_x, delta_y, delta_z, the step for finite
    difference method gradients. We appromate the time derivative at t
    (1rst step rk4) with a forward finite difference and the time derivative
    at t+delta_t (4th step rk4) with a backward finite difference.
    """
    Bterm_tau = 2 * (1 - particle.B) / (1. + 2 * particle.B) * particle.tau
    w0 = (2 * (1 - particle.B) / (1. + 2 * particle.B)
          * fieldset.g * particle.tau)
    w0tau = w0 * particle.tau

    norm_deltax = 1.0 / (2.0 * fieldset.delta_x)
    norm_deltay = 1.0 / (2.0 * fieldset.delta_y)
    norm_deltaz = 1.0 / (2.0 * fieldset.delta_z)
    norm_deltat = 1.0 / (1.0 * particle.dt)

    # RK4 STEP 1
    # fluid field velocity at location of particle
    (uf1, vf1, wf1) = fieldset.UVW[time, particle.depth,
                                   particle.lat, particle.lon]

    # calculate time derivative of fluid field
    (uf_tp1, vf_tp1, wf_tp1) = fieldset.UVW[time+particle.dt, particle.depth,
                                            particle.lat, particle.lon]
    (uf_tm1, vf_tm1, wf_tm1) = fieldset.UVW[time, particle.depth,
                                            particle.lat, particle.lon]
    dudt1 = (uf_tp1 - uf_tm1) * norm_deltat
    dvdt1 = (vf_tp1 - vf_tm1) * norm_deltat
    dwdt1 = (wf_tp1 - wf_tm1) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm1, v_dxm1, w_dxm1) = fieldset.UVW[time, particle.depth,
                                            particle.lat,
                                            particle.lon-fieldset.delta_x]
    (u_dxp1, v_dxp1, w_dxp1) = fieldset.UVW[time, particle.depth,
                                            particle.lat,
                                            particle.lon + fieldset.delta_x]
    (u_dym1, v_dym1, w_dym1) = fieldset.UVW[time, particle.depth,
                                            particle.lat-fieldset.delta_y,
                                            particle.lon]
    (u_dyp1, v_dyp1, w_dyp1) = fieldset.UVW[time, particle.depth,
                                            particle.lat+fieldset.delta_y,
                                            particle.lon]
    (u_dzm1, v_dzm1, w_dzm1) = fieldset.UVW[time,
                                            particle.depth - fieldset.delta_z,
                                            particle.lat, particle.lon]
    (u_dzp1, v_dzp1, w_dzp1) = fieldset.UVW[time,
                                            particle.depth + fieldset.delta_z,
                                            particle.lat, particle.lon]
    dudx1 = (u_dxp1 - u_dxm1) * norm_deltax
    dudy1 = (u_dyp1 - u_dym1) * norm_deltay
    dudz1 = (u_dzp1 - u_dzm1) * norm_deltaz
    dvdx1 = (v_dxp1 - v_dxm1) * norm_deltax
    dvdy1 = (v_dyp1 - v_dym1) * norm_deltay
    dvdz1 = (v_dzp1 - v_dzm1) * norm_deltaz
    dwdx1 = (w_dxp1 - w_dxm1) * norm_deltax
    dwdy1 = (w_dyp1 - w_dym1) * norm_deltay
    dwdz1 = (w_dzp1 - w_dzm1) * norm_deltaz

    # caluclate material derivative fluid
    DuDt1 = dudt1 + uf1 * dudx1 + vf1 * dudy1 + wf1 * dudz1
    DvDt1 = dvdt1 + uf1 * dvdx1 + vf1 * dvdy1 + wf1 * dvdz1
    DwDt1 = dwdt1 + uf1 * dwdx1 + vf1 * dwdy1 + wf1 * dwdz1

    # coriolis force
    f1 = 0  # 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor1 = -vf1 * f1
    vcor1 = uf1 * f1

    u1 = uf1 + Bterm_tau * (DuDt1 + ucor1) - w0tau * dudz1
    v1 = vf1 + Bterm_tau * (DvDt1 + vcor1) - w0tau * dvdz1
    w1 = wf1 + w0 + Bterm_tau * DwDt1 - w0tau * dwdz1

    # lon, lat for next step
    lon1 = particle.lon + 0.5 * u1 * particle.dt
    lat1 = particle.lat + 0.5 * v1 * particle.dt
    depth1 = particle.depth + 0.5 * w1 * particle.dt
    time1 = time + 0.5 * particle.dt

    # RK4 STEP 2
    # fluid field velocity at location of particle
    (uf2, vf2, wf2) = fieldset.UVW[time1, depth1, lat1, lon1]

    # calculate time derivative of fluid field
    (uf_tp2, vf_tp2, wf_tp2) = fieldset.UVW[time+particle.dt,
                                            depth1, lat1, lon1]
    (uf_tm2, vf_tm2, wf_tm2) = fieldset.UVW[time1, depth1, lat1, lon1]
    dudt2 = (uf_tp2 - uf_tm2) * norm_deltat
    dvdt2 = (vf_tp2 - vf_tm2) * norm_deltat
    dwdt2 = (wf_tp2 - wf_tm2) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm2, v_dxm2, w_dxm2) = fieldset.UVW[time1, depth1,
                                            lat1, lon1-fieldset.delta_x]
    (u_dxp2, v_dxp2, w_dxp2) = fieldset.UVW[time1, depth1,
                                            lat1, lon1+fieldset.delta_x]
    (u_dym2, v_dym2, w_dym2) = fieldset.UVW[time1, depth1,
                                            lat1-fieldset.delta_y, lon1]
    (u_dyp2, v_dyp2, w_dyp2) = fieldset.UVW[time1, depth1,
                                            lat1+fieldset.delta_y, lon1]
    (u_dzm2, v_dzm2, w_dzm2) = fieldset.UVW[time1, depth1-fieldset.delta_z,
                                            lat1, lon1]
    (u_dzp2, v_dzp2, w_dzp2) = fieldset.UVW[time1, depth1+fieldset.delta_z,
                                            lat1, lon1]
    dudx2 = (u_dxp2 - u_dxm2) * norm_deltax
    dudy2 = (u_dyp2 - u_dym2) * norm_deltay
    dudz2 = (u_dzp2 - u_dzm2) * norm_deltaz
    dvdx2 = (v_dxp2 - v_dxm2) * norm_deltax
    dvdy2 = (v_dyp2 - v_dym2) * norm_deltay
    dvdz2 = (v_dzp2 - v_dzm2) * norm_deltaz
    dwdx2 = (w_dxp2 - w_dxm2) * norm_deltax
    dwdy2 = (w_dyp2 - w_dym2) * norm_deltay
    dwdz2 = (w_dzp2 - w_dzm2) * norm_deltaz

    # caluclate material derivative fluid
    DuDt2 = dudt2 + uf2 * dudx2 + vf2 * dudy2 + wf2 * dudz2
    DvDt2 = dvdt2 + uf2 * dvdx2 + vf2 * dvdy2 + wf2 + dvdz2
    DwDt2 = dwdt2 + uf2 * dwdx2 + vf2 * dwdy2 + wf2 + dwdz2

    # coriolis force
    f2 = 0  # 2 * fieldset.Omega_earth*math.sin(lat1 * math.pi / 180)
    ucor2 = -vf2 * f2
    vcor2 = uf2 * f2

    u2 = uf2 + Bterm_tau * (DuDt2 + ucor2) - w0tau * dudz2
    v2 = vf2 + Bterm_tau * (DvDt2 + vcor2) - w0tau * dvdz2
    w2 = wf2 + w0 + Bterm_tau * DwDt2 - w0tau * dwdz2

    # lon, lat for next step
    lon2 = particle.lon + 0.5 * u2 * particle.dt
    lat2 = particle.lat + 0.5 * v2 * particle.dt
    depth2 = particle.depth + 0.5 * w2 * particle.dt
    time2 = time + 0.5 * particle.dt

    # RK4 STEP 3
    # fluid field velocity at location of particle
    (uf3, vf3, wf3) = fieldset.UVW[time2, depth2, lat2, lon2]

    # calculate time derivative of fluid field
    (uf_tp3, vf_tp3, wf_tp3) = fieldset.UVW[time+particle.dt,
                                            depth2, lat2, lon2]
    (uf_tm3, vf_tm3, wf_tm3) = fieldset.UVW[time, depth2, lat2, lon2]
    dudt3 = (uf_tp3 - uf_tm3) * norm_deltat
    dvdt3 = (vf_tp3 - vf_tm3) * norm_deltat
    dwdt3 = (wf_tp3 - wf_tm3) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm3, v_dxm3, w_dxm3) = fieldset.UVW[time2, depth2,
                                            lat2, lon2-fieldset.delta_x]
    (u_dxp3, v_dxp3, w_dxp3) = fieldset.UVW[time2, depth2,
                                            lat2, lon2+fieldset.delta_x]
    (u_dym3, v_dym3, w_dym3) = fieldset.UVW[time2, depth2,
                                            lat2-fieldset.delta_y, lon2]
    (u_dyp3, v_dyp3, w_dyp3) = fieldset.UVW[time2, depth2,
                                            lat2+fieldset.delta_y, lon2]
    (u_dzm3, v_dzm3, w_dzm3) = fieldset.UVW[time2, depth2-fieldset.delta_z,
                                            lat2, lon2]
    (u_dzp3, v_dzp3, w_dzp3) = fieldset.UVW[time2, depth2+fieldset.delta_z,
                                            lat2, lon2]
    dudx3 = (u_dxp3 - u_dxm3) * norm_deltax
    dudy3 = (u_dyp3 - u_dym3) * norm_deltay
    dudz3 = (u_dzp3 - u_dzm3) * norm_deltaz
    dvdx3 = (v_dxp3 - v_dxm3) * norm_deltax
    dvdy3 = (v_dyp3 - v_dym3) * norm_deltay
    dvdz3 = (v_dzp3 - v_dzm3) * norm_deltaz
    dwdx3 = (w_dxp3 - w_dxm3) * norm_deltax
    dwdy3 = (w_dyp3 - w_dym3) * norm_deltay
    dwdz3 = (w_dzp3 - w_dzm3) * norm_deltaz

    # caluclate material derivative fluid
    DuDt3 = dudt3 + uf3 * dudx3 + vf3 * dudy3 + wf3 * dudz3
    DvDt3 = dvdt3 + uf3 * dvdx3 + vf3 * dvdy3 + wf3 + dvdz3
    DwDt3 = dwdt3 + uf3 * dwdx3 + vf3 * dwdy3 + wf3 + dwdz3

    # coriolis force
    f3 = 0  # 2 * fieldset.Omega_earth*math.sin(lat2 * math.pi / 180)
    ucor3 = -vf3 * f3
    vcor3 = uf3 * f3

    u3 = uf3 + Bterm_tau * (DuDt3 + ucor3) - w0tau * dudz3
    v3 = vf3 + Bterm_tau * (DvDt3 + vcor3) - w0tau * dvdz3
    w3 = wf3 + w0 + Bterm_tau * DwDt3 - w0tau * dwdz3

    # lon, lat for next step
    lon3 = particle.lon + u3 * particle.dt
    lat3 = particle.lat + v3 * particle.dt
    depth3 = particle.depth + w3 * particle.dt
    time3 = time + particle.dt

    # RK4 STEP 4
    # fluid field velocity at location of particle
    (uf4, vf4, wf4) = fieldset.UVW[time3, depth3, lat3, lon3]

    # calculate time derivative of fluid field
    (uf_tp4, vf_tp4, wf_tp4) = fieldset.UVW[time+particle.dt,
                                            depth3, lat3, lon3]
    (uf_tm4, vf_tm4, wf_tm4) = fieldset.UVW[time, depth3, lat3, lon3]
    dudt4 = (uf_tp4 - uf_tm4) * norm_deltat
    dvdt4 = (vf_tp4 - vf_tm4) * norm_deltat
    dwdt4 = (wf_tp4 - wf_tm4) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm4, v_dxm4, w_dxm4) = fieldset.UVW[time3, depth3,
                                            lat3, lon3-fieldset.delta_x]
    (u_dxp4, v_dxp4, w_dxp4) = fieldset.UVW[time3, depth3,
                                            lat3, lon3+fieldset.delta_x]
    (u_dym4, v_dym4, w_dym4) = fieldset.UVW[time3, depth3,
                                            lat3-fieldset.delta_y, lon3]
    (u_dyp4, v_dyp4, w_dyp4) = fieldset.UVW[time3, depth3,
                                            lat3+fieldset.delta_y, lon3]
    (u_dzm4, v_dzm4, w_dzm4) = fieldset.UVW[time3, depth3-fieldset.delta_z,
                                            lat3, lon3]
    (u_dzp4, v_dzp4, w_dzp4) = fieldset.UVW[time3, depth3+fieldset.delta_z,
                                            lat3, lon3]
    dudx4 = (u_dxp4 - u_dxm4) * norm_deltax
    dudy4 = (u_dyp4 - u_dym4) * norm_deltay
    dudz4 = (u_dzp4 - u_dzm4) * norm_deltaz
    dvdx4 = (v_dxp4 - v_dxm4) * norm_deltax
    dvdy4 = (v_dyp4 - v_dym4) * norm_deltay
    dvdz4 = (v_dzp4 - v_dzm4) * norm_deltaz
    dwdx4 = (w_dxp4 - w_dxm4) * norm_deltax
    dwdy4 = (w_dyp4 - w_dym4) * norm_deltay
    dwdz4 = (w_dzp4 - w_dzm4) * norm_deltaz

    # calcuclate material derivative fluid
    DuDt4 = dudt4 + uf4 * dudx4 + vf4 * dudy4 + wf4 * dudz4
    DvDt4 = dvdt4 + uf4 * dvdx4 + vf4 * dvdy4 + wf4 * dvdz4
    DwDt4 = dwdt4 + uf4 * dwdx4 + vf4 * dwdy4 + wf4 * dwdz4

    # coriolis force
    f4 = 0   # 2 * fieldset.Omega_earth * math.sin(lat3 * math.pi / 180)
    ucor4 = -vf4 * f4
    vcor4 = uf4 * f4

    u4 = uf4 + Bterm_tau * (DuDt4 + ucor4) - w0tau * dudz4
    v4 = vf4 + Bterm_tau * (DvDt4 + vcor4) - w0tau * dvdz4
    w4 = wf4 + w0 + Bterm_tau * DwDt4 - w0tau * dwdz4

    # RK4 INTEGRATION STEP
    particle_dlon += (u1 + 2 * u2 + 2 * u3 + u4) * particle.dt / 6.0
    particle_dlat += (v1 + 2 * v2 + 2 * v3 + v4) * particle.dt / 6.0
    particle_ddepth += (w1 + 2 * w2 + 2 * w3 + w4) * particle.dt / 6.0
