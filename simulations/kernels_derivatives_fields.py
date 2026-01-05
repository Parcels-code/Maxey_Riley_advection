from parcels import JITParticle, Variable
import math
import numpy as np


def MRAdvectionRK4_2D_drag_Rep_derivatives_field(particle, fieldset, time):
    """
    Advection of particles using Maxey-Riley equation in 2D without Basset
    history term and Faxen corrections without sinking or floating force.
    The equation is numerically integrated using the 4th order runge kutta
    scheme for a 2nd order ODE equation. This function uses precomputed 
    derivative fields to calculate the gradients in the flow.
    We use the a fluid drag that depends on the particle reynolds number.
    We use the equation given by morrision 2013 which is valid up to Re_p= 10^6. 



    dependencies:
    - up, vp, particle velocity (particle variables)
    - B, buoyancy particle (particle variable)
    - Omega_earth, angular velocity earth (fieldset constant)
    - nu, kinematic velocity (fieldset constant) [m/s^2]
    - d, diameter particle, (particle constant)  [m]
    """ 
    Bterm = 3. / (1. + 2. * particle.B)
    Bterm2 = 2 * (1 - particle.B) / (1 + 2 * particle.B) 

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
    dudt1 = fieldset.dudt[time,particle.depth, particle.lat, particle.lon]
    dvdt1 = fieldset.dvdt[time,particle.depth, particle.lat, particle.lon]


    # calculate spatial gradients fluid field
    dudx1 = fieldset.dudx[time,particle.depth, particle.lat, particle.lon]
    dvdx1 = fieldset.dvdx[time,particle.depth, particle.lat, particle.lon]
    dudy1 = fieldset.dudy[time,particle.depth, particle.lat, particle.lon]
    dvdy1 = fieldset.dvdy[time,particle.depth, particle.lat, particle.lon]

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
    dudt2 = fieldset.dudt[time1, particle.depth, lat1,lon1]
    dvdt2 = fieldset.dvdt[time1, particle.depth, lat1,lon1]

    # calculate spatial gradients fluid field
    dudx2 = fieldset.dudx[time1, particle.depth, lat1,lon1]
    dvdx2 = fieldset.dvdx[time1, particle.depth, lat1,lon1]
    dudy2 = fieldset.dudy[time1, particle.depth, lat1,lon1]
    dvdy2 = fieldset.dvdy[time1, particle.depth, lat1,lon1]

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
    dudt3 = fieldset.dudt[time2, particle.depth, lat2,lon2]
    dvdt3 = fieldset.dvdt[time2, particle.depth, lat2,lon2]

    # calculate spatial gradients fluid field
    dudx3 = fieldset.dudx[time2, particle.depth, lat2,lon2]
    dvdx3 = fieldset.dvdx[time2, particle.depth, lat2,lon2]
    dudy3 = fieldset.dudy[time2, particle.depth, lat2,lon2]
    dvdy3 = fieldset.dvdy[time2, particle.depth, lat2,lon2]

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
    dudt4 = fieldset.dudt[time3, particle.depth, lat3,lon3]
    dvdt4 = fieldset.dvdt[time3, particle.depth, lat3,lon3]

    # calculate spatial gradients fluid field
    dudx4 = fieldset.dudx[time3, particle.depth, lat3,lon2]
    dvdx4 = fieldset.dvdx[time3, particle.depth, lat3,lon3]
    dudy4 = fieldset.dudy[time3, particle.depth, lat3,lon3]
    dvdy4 = fieldset.dvdy[time3, particle.depth, lat3,lon3]

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

def MRAdvectionRK4_2D_drag_Rep_constant_derivatives_field(particle, fieldset, time):
    """
    Advection of particles using Maxey-Riley equation in 2D without Basset
    history term and Faxen corrections without sinking or floating force.
    The equation is numerically integrated using the 4th order runge kutta
    scheme for a 2nd order ODE equation. we use precomputed fieldsets
    for the derivatives


    dependencies:
    - up, vp, particle velocity (particle variables)
    - B, buoyancy particle (particle variable)
    - Omega_earth, angular velocity earth (fieldset constant)
    - tau stokes relaxatioin time (particle variable)
    - C_Rep corection factor viscosity for Rep > 1
    - d, diameter particle, (particle constant)  [m]
      (fieldset constants)
    """ 
    Bterm = 3. / (1. + 2. * particle.B)
    Bterm2 = 2 * (1 - particle.B) / (1 + 2 * particle.B) 
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
    dudt1 = fieldset.dudt[time,particle.depth,particle.lat,particle.lon]
    dvdt1 = fieldset.dvdt[time,particle.depth,particle.lat,particle.lon]

    # calculate spatial gradients fluid field
    dudx1 = fieldset.dudx[time,particle.depth,particle.lat,particle.lon]
    dvdx1 = fieldset.dvdx[time,particle.depth,particle.lat,particle.lon]
    dudy1 = fieldset.dudy[time,particle.depth,particle.lat,particle.lon]
    dvdy1 = fieldset.dvdy[time,particle.depth,particle.lat,particle.lon]

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
    dudt2 = fieldset.dudt[time1, particle.depth, lat1, lon1]
    dvdt2 = fieldset.dvdt[time1, particle.depth, lat1, lon1]

    # calculate spatial gradients fluid field
    dudx2 = fieldset.dudx[time1, particle.depth, lat1, lon1]
    dvdx2 = fieldset.dvdx[time1, particle.depth, lat1, lon1]
    dudy2 = fieldset.dudy[time1, particle.depth, lat1, lon1]
    dvdy2 = fieldset.dvdy[time1, particle.depth, lat1, lon1]

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
    dudt3 = fieldset.dudt[time2, particle.depth, lat2, lon2]
    dvdt3 = fieldset.dvdt[time2, particle.depth, lat2, lon2]

    # calculate spatial gradients fluid field
    dudx3 = fieldset.dudx[time2, particle.depth, lat2, lon2]
    dvdx3 = fieldset.dvdx[time2, particle.depth, lat2, lon2]
    dudy3 = fieldset.dudy[time2, particle.depth, lat2, lon2]
    dvdy3 = fieldset.dvdy[time2, particle.depth, lat2, lon2]

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
    dudt4 = fieldset.dudt[time3, particle.depth, lat3, lon3]
    dvdt4 = fieldset.dvdt[time3, particle.depth, lat3, lon3]

    # calculate spatial gradients fluid field
    dudx4 = fieldset.dudx[time3, particle.depth, lat3, lon3]
    dvdx4 = fieldset.dvdx[time3, particle.depth, lat3, lon3]
    dudy4 = fieldset.dudy[time3, particle.depth, lat3, lon3]
    dvdy4 = fieldset.dvdy[time3, particle.depth, lat3, lon3]


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
 
def MRSMAdvectionRK4_2D_drag_Rep_derivatives_field(particle, fieldset, time):
    """
    Advection of particles using the slow manifold approximation of the Maxey-
    Riley equation in 2D without Basset history term and Faxen corrections. The
    equation is numerically integrated using the 4th order runge kutta scheme
    for an overdamped equation (ODE order 1). This function uses fieldsets for
    derivatives that are precomputed. 
    dependencies:
    - up, vp, particle velocity (particle variables)
    - tau, stokes relaxation time particle (particle variable)
    - B, buoyancy particle (particle variable)
    - Omega_earth, angular velocity earth (fieldset constant)
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
    
    # RK4 STEP 1
    # fluid field velocity at location of particle


    
    if(fieldset.gradient == False):
        DuDt1 = 0
        DvDt1 = 0
    else:
        # calculate time derivative of fluid field
        dudt1 = fieldset.dudt[time,particle.depth,particle.lat,particle.lon]
        dvdt1 = fieldset.dvdt[time,particle.depth,particle.lat,particle.lon]

        # calculate spatial gradients fluid field
        dudx1 = fieldset.dudx[time,particle.depth,particle.lat,particle.lon]
        dvdx1 = fieldset.dvdx[time,particle.depth,particle.lat,particle.lon]
        dudy1 = fieldset.dudy[time,particle.depth,particle.lat,particle.lon]
        dvdy1 = fieldset.dvdy[time,particle.depth,particle.lat,particle.lon]

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
        dudt2 = fieldset.dudt[time1,particle.depth, lat1, lon1]
        dvdt2 = fieldset.dvdt[time1,particle.depth, lat1, lon1]

        # calculate spatial gradients fluid field
        dudx2 = fieldset.dudx[time1,particle.depth, lat1, lon1]
        dvdx2 = fieldset.dvdx[time1,particle.depth, lat1, lon1]
        dudy2 = fieldset.dudy[time1,particle.depth, lat1, lon1]
        dvdy2 = fieldset.dvdy[time1,particle.depth, lat1, lon1]

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
        dudt3 = fieldset.dudt[time2,particle.depth, lat2, lon2]
        dvdt3 = fieldset.dvdt[time2,particle.depth, lat2, lon2]

        # calculate spatial gradients fluid field
        dudx3 = fieldset.dudx[time2,particle.depth, lat2, lon2]
        dvdx3 = fieldset.dvdx[time2,particle.depth, lat2, lon2]
        dudy3 = fieldset.dudy[time2,particle.depth, lat2, lon2]
        dvdy3 = fieldset.dvdy[time2,particle.depth, lat2, lon2]

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
        dudt4 = fieldset.dudt[time3,particle.depth, lat3, lon3]
        dvdt4 = fieldset.dvdt[time3,particle.depth, lat3, lon3]

        # calculate spatial gradients fluid field
        dudx4 = fieldset.dudx[time3,particle.depth, lat3, lon3]
        dvdx4 = fieldset.dvdx[time3,particle.depth, lat3, lon3]
        dudy4 = fieldset.dudy[time3,particle.depth, lat3, lon3]
        dvdy4 = fieldset.dvdy[time3,particle.depth, lat3, lon3]
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


def MRSMAdvectionRK4_2D_drag_Rep_constant_derivatives_field(particle, fieldset, time):
    """
    Advection of particles using the slow manifold approximation of the Maxey-
    Riley equation in 2D without Basset history term and Faxen corrections. The
    equation is numerically integrated using the 4th order runge kutta scheme
    for an overdamped equation (ODE order 1). We use precomputed fieldsets
    for the derivatives

    dependencies:
    - up, vp, particle velocity (particle variables)
    - tau, stokes relaxation time particle (particle variable)
    - B, buoyancy particle (particle variable)
    - Omega_earth, angular velocity earth (fieldset constant)
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

    
    # RK4 STEP 1
    # fluid field velocity at location of particle
    (uf1, vf1) = fieldset.UV[time, particle.depth,
                             particle.lat, particle.lon]
    if(fieldset.gradient == False):
        DuDt1 = 0
        DvDt1 = 0
    else:
        # calculate time derivative of fluid field
        dudt1= fieldset.dudt[time,particle.depth,particle.lat,particle.lon]
        dvdt1= fieldset.dvdt[time,particle.depth,particle.lat,particle.lon]

        # calculate spatial gradients fluid field
        dudx1= fieldset.dudx[time,particle.depth,particle.lat,particle.lon]
        dvdx1= fieldset.dvdx[time,particle.depth,particle.lat,particle.lon]
        dudy1= fieldset.dudy[time,particle.depth,particle.lat,particle.lon]
        dvdy1= fieldset.dvdy[time,particle.depth,particle.lat,particle.lon]

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
        dudt2= fieldset.dudt[time1,particle.depth,lat1,lon1]
        dvdt2= fieldset.dvdt[time1,particle.depth,lat1,lon1]

        # calculate spatial gradients fluid field
        dudx2= fieldset.dudx[time1,particle.depth,lat1,lon1]
        dvdx2= fieldset.dvdx[time1,particle.depth,lat1,lon1]
        dudy2= fieldset.dudy[time1,particle.depth,lat1,lon1]
        dvdy2= fieldset.dvdy[time1,particle.depth,lat1,lon1]

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
        dudt3= fieldset.dudt[time2,particle.depth,lat2,lon2]
        dvdt3= fieldset.dvdt[time2,particle.depth,lat2,lon2]

        # calculate spatial gradients fluid field
        dudx3= fieldset.dudx[time2,particle.depth,lat2,lon2]
        dvdx3= fieldset.dvdx[time2,particle.depth,lat2,lon2]
        dudy3= fieldset.dudy[time2,particle.depth,lat2,lon2]
        dvdy3= fieldset.dvdy[time2,particle.depth,lat2,lon2]

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
        dudt4= fieldset.dudt[time3,particle.depth,lat3,lon3]
        dvdt4= fieldset.dvdt[time3,particle.depth,lat3,lon3]

        # calculate spatial gradients fluid field
        dudx4= fieldset.dudx[time3,particle.depth,lat3,lon3]
        dvdx4= fieldset.dvdx[time3,particle.depth,lat3,lon3]
        dudy4= fieldset.dudy[time3,particle.depth,lat3,lon3]
        dvdy4= fieldset.dvdy[time3,particle.depth,lat3,lon3]

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
