from parcels import JITParticle, Variable
import math
import numpy as np


def MRAdvectionRK4_2D_Newtonian_drag(particle, fieldset, time):
    """
    Advection of particles using Maxey-Riley equation in 2D without Basset
    history term and Faxen corrections without sinking or floating force.
    We use the newtonian drag formulation for particle reynolds
    numbers > 3000. Note that we work with lengthscale here (tau now represents
    lengthscale which is given in meters and should be converted to degrees in 
    the calculation)
    The equation is numerically integrated using the 4th order runge kutta
    scheme for a 2nd order ODE equation. We appromate the time derivative at t 
    (1rst step rk4) with a forward finite difference and the time derivative
    at t+delta_t (4th step rk4) with a backward finite difference.

    dependencies:
    - up, vp, particle velocity (particle variables)
    - tau, drag length scale particle in m (particle variable)
    - B, buoyancy particle (particle variable)
    - Omega_earth, angular velocity earth (fieldset constant)
    - delta_x, delta_y, delta_t step for finite difference method gradients 
      (fieldset constants)
    """
    ld_degrees_inv =  math.pi * fieldset.Rearth / (180 * particle.tau)
    Bterm = 3. / (1. + 2. * particle.B)
    Bterm2 = 2 * (1 - particle.B) / (1 + 2 * particle.B) 
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

    # coriolis force
    f1 = 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor1 = -(vf1 - vp1) * f1
    vcor1 = (uf1 - up1) * f1
    upcor1 = -vp1 * f1
    vpcor1 = up1 * f1

    #lift force
    omega1 = dvdx1 - dudy1
    ulift1 =  -(vf1 - vp1) * omega1
    vlift1 = (uf1 - up1) * omega1

  
    # Newtonian drag force
    uslip_mag1 = math.sqrt((uf1 - up1)**2 + (vf1 - vp1)**2)
    udrag1 = ld_degrees_inv * math.cos(particle.lat * math.pi / 180)  * uslip_mag1 * (uf1 - up1)
    vdrag1 = ld_degrees_inv * uslip_mag1 * (vf1 - vp1)

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

    # coriolis force
    f2 = 2 * fieldset.Omega_earth * math.sin(lat1 * math.pi / 180.)
    ucor2 = -(vf2 - vp2) * f2
    vcor2 = (uf2 - up2) * f2
    upcor2 = -vp2 * f2
    vpcor2 = up2 * f2

    #lift force
    omega2 = dvdx2 - dudy2
    ulift2 =  -(vf2 - vp2) * omega2
    vlift2 = (uf2 - up2) * omega2

    # Newtonian drag force
    uslip_mag2 = math.sqrt((uf2 - up2)**2 + (vf2 - vp2)**2)
    udrag2 = ld_degrees_inv * math.cos(lat1 * math.pi / 180) * uslip_mag2 * (uf2 - up2)
    vdrag2 = ld_degrees_inv * uslip_mag2 * (vf2 - vp2)

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

    # coriolis force
    f3 = 2 * fieldset.Omega_earth * math.sin(lat2 * math.pi / 180.)
    ucor3 = -(vf3 - vp3) * f3
    vcor3 = (uf3 - up3) * f3
    upcor3 = -vp3 * f3
    vpcor3 = up3 * f3

    #lift force
    omega3 = dvdx3 - dudy3
    ulift3 =  -(vf3 - vp3) * omega3
    vlift3 = (uf3 - up3) * omega3

    # Newtonian drag force
    uslip_mag3 = math.sqrt((uf3 - up3)**2 + (vf3 - vp3)**2)
    udrag3 = ld_degrees_inv * math.cos(lat2 * math.pi / 180) * uslip_mag3 * (uf3 - up3)
    vdrag3 = ld_degrees_inv * uslip_mag3 * (vf3 - vp3)

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

    # coriolis force
    f4 =  2 * fieldset.Omega_earth * math.sin(lat3 * math.pi / 180)
    ucor4 = -(vf4 - vp4) * f4
    vcor4 = (uf4 - up4) * f4
    upcor4 = -vp4 * f4
    vpcor4 = up4 * f4

    #lift force
    omega4 = dvdx4 - dudy4
    ulift4 =  -(vf4 - vp4) * omega4
    vlift4 = (uf4 - up4) * omega4

    # Newtonian drag force
    uslip_mag4 = math.sqrt((uf4 - up4)**2 + (vf4 - vp4)**2)
    udrag4 = ld_degrees_inv * math.cos(lat3 * math.pi / 180)  * uslip_mag4 * (uf4 - up4)
    vdrag4 = ld_degrees_inv * uslip_mag4 * (vf4 - vp4)

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
