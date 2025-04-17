"""
Functions to calculate particle charactaristics like
- stokes kinematic_viscositymber
- particle reynolds kinematic_viscositymber
- drag coefficient correction
- buoyancy
- ...
"""

# import needed backages
import numpy as np
import xarray as xr


def buoyancy_drifter(
    diameter: float, heigth: float, mass: float, density_fluid: float
) -> float:
    volume_drifter = 1 / 4 * np.pi * diameter * diameter * heigth
    density_drifter = mass / volume_drifter
    buoyancy = density_drifter / density_fluid
    return buoyancy


def stokes_relaxation_time(
    diameter: float, kinematic_viscosity: float, buoyancy: float
) -> float:
    """
    Calculates stokes relaxation time for MR equation
    - d: diameter [m]
    - kinematic_viscosity kinematic viscoscity sea water [m^2/s]
    - Buoyancy particle (rho_p / rho_w) [unitless]
    """
    return (1 + 2 * buoyancy) * diameter * diameter / (36 * kinematic_viscosity)


def Re_particle(Uslip: float, diameter: float, kinematic_viscosity: float) -> float:
    return np.abs(Uslip) * diameter / kinematic_viscosity


def diffusion_time(diameter: float, kinematic_viscosity: float) -> float:
    """
    -  diamater particle [m]
    -  kinematic viscosity water [m^2/s]
    """
    return diameter * diameter / kinematic_viscosity


def factor_drag_white1991(Rep: float) -> float:
    """
    stokes drag corection factor valid up to Rep < 10^5 from
    [1] F. M. White, Viscous fluid flow 2nd edition (1991)(p. 182)
    dependencies:
    - Rep : particle reynolds kinematic_viscositymber
    """
    c_REp = 1 + Rep / (4.0 * (1 + np.sqrt(Rep))) + Rep / 60.0
    return c_REp


def factor_drag_morrison2013(Rep: float) -> float:
    """
    Emperical stokes drag correction factor valid up to Rep < 10^6 from
    [1] F. A. Morrison, An introduction to fluid mechanics (2013)
    dependencies:
    - Rep : particle reynolds kinematic_viscositymber
    """
    c_REp = (
        1
        + Rep / 24 * 2.6 * (Rep / 5.0) / (1 + (Rep / 5.0) ** (1.52))
        + Rep
        / 24
        * 0.411
        * (Rep / (2.63 * 10**5)) ** (-7.94)
        / (1 + (Rep / (2.63 * 10**5)) ** (-8))
        + Rep / 24 * 0.25 * (Rep) / ((10**6) + Rep)
    )

    return c_REp


def factor_drag_Schiller1933(Rep: float) -> float:
    """
    Emperical stokes drag correction factor valid up to Rep < 800 from
    [1] Schiller 1933
    dependencies:
    - Rep : particle reynolds kinematic_viscositymber
    """
    c_REp = 1 + 0.15 * Rep ** (0.687)

    return c_REp


def drag_length(
    diameter: float, buoyancy: float, drag_coefficient: float = 0.45
) -> float:
    """ "
    prefactor for newtonian drag
    dependencies:
    - d: diameter particle [m]
    - B: Buoyancy (rho_p/rho_f)
    - Cd: drag coefficient =0.45 for smooth sphere

    """
    return 2 * (1 + 2 * buoyancy) * diameter / (3 * drag_coefficient)


def find_rep_white1991(
    Rep: float,
    tau_coriolis: float,
    tau_tides: float,
    buoyancy: float,
    kinematic_viscosity: float,
    uf: float,
    diameter: float,
) -> float:
    """
    funtion needed to find Reynolds kinematic_viscositymber (using find root)
    using the slow manifold maxey riley equation. To make sure
    that the input REp is same as the calculated Rep.
    Drag coefficient calculated using white1991
    dependencies
    - Rep: input particle Reynolds kinematic_viscositymber
    - tau_coriolis: coriolis time (2 Omega_earth sin(lat))^-1 [1/s]
    - buoyancy particle
    - kinematic_viscosity: kinematic viscosisty water [m^2/s]
    - uf: flow velocity [m/s]
    - d: diameter particle
    """
    tau_stokes = stokes_relaxation_time(diameter, kinematic_viscosity, buoyancy)
    Bterm = 2 * (1 - buoyancy) / (1 + 2 * buoyancy)
    uslip = tau_stokes * Bterm * (1 / tau_coriolis + 1 / tau_tides)
    Rep_calculated = Re_particle(uslip, diameter, kinematic_viscosity)
    return Rep * factor_drag_white1991(Rep) - Rep_calculated


def find_rep_morrison2013(
    Rep: float,
    tau_coriolis: float,
    tau_tides: float,
    buoyancy: float,
    kinematic_viscosity: float,
    diameter: float,
) -> float:
    """
    funtion needed to find Reynolds kinematic_viscositymber (using find root)
    using the slow manifold maxey riley equation. To make sure
    that the input REp is same as the calculated Rep.
    Drag coefficient calculated using morrison2013
    dependencies
    - Rep: input particle Reynolds kinematic_viscositymber
    - tau_coriolis: coriolis time (2 Omega_earth sin(lat))^-1 [1/s]
    - B: buoyancy particle
    - kinematic_viscosity: kinematic viscosisty water [m^2/s]
    - d: diameter particle
    """
    tau_stokes = stokes_relaxation_time(diameter, kinematic_viscosity, buoyancy)
    Bterm = 2 * (1 - buoyancy) / (1 + 2 * buoyancy)
    uslip = tau_stokes * Bterm * (1 / tau_coriolis + 1 / tau_tides)
    Rep_calculated = Re_particle(uslip, diameter, kinematic_viscosity)
    return Rep * factor_drag_morrison2013(Rep) - Rep_calculated


def find_rep_schiller1933(
    Rep: float,
    tau_coriolis: float,
    tau_tides: float,
    buoyancy: float,
    kinematic_viscosity: float,
    diameter: float,
) -> float:
    """
    funtion needed to find Reynolds kinematic_viscositymber (using find root)
    using the slow manifold maxey riley equation. To make sure
    that the input REp is same as the calculated Rep.
    Drag coefficient calculated using Schiller1933
    dependencies
    - Rep: input particle Reynolds kinematic_viscositymber
    - tau_coriolis: coriolis time (2 Omega_earth sin(lat))^-1 [1/s]
    - B: buoyancy particle
    - kinematic_viscosity: kinematic viscosisty water [m^2/s]
    - uf: flow velocity [m/s]
    - d: diameter particle
    """
    tau_stokes = stokes_relaxation_time(diameter, kinematic_viscosity, buoyancy)
    Bterm = 2 * (1 - buoyancy) / (1 + 2 * buoyancy)
    uslip = tau_stokes * Bterm * (1 / tau_coriolis + 1 / tau_tides)
    Rep_calculated = Re_particle(uslip, diameter, kinematic_viscosity)
    return Rep * factor_drag_Schiller1933(Rep) - Rep_calculated


def slip_force(u, d, nu, rho, Rep):
    C_Rep = factor_drag_white1991(Rep)
    return u * 3 * np.pi * C_Rep * nu * d * rho


# def coriolis_force(u,d,nu,rho,Rep):
def measure_coriolis_force(particle, fieldset, time):
    """f
    Kernel to measure the coriolis force (2D) of the fluid at the location
    of the particle
    """
    Bterm = 3.0 / (1.0 + 2.0 * particle.B)
    Bterm2 = 2 * (1 - particle.B) / (1 + 2 * particle.B)
    uf, vf = fieldset.UV[time, particle.depth, particle.lat, particle.lon]

    # coriolis force
    f = 2 * fieldset.Omega_earth * math.sin(particle.lat * math.pi / 180)
    ucor = -(vf - particle.vp) * f
    vcor = (uf - particle.up) * f
    upcor = -particle.vp * f
    vpcor = particle.up * f
    fcor_u = Bterm * ucor + Bterm2 * upcor
    fcor_v = Bterm + vcor + Bterm2 * vpcor

    particle.fcor_u = (
        fcor_u
        * fieldset.Rearth
        * math.cos(particle.lat * math.pi / 180.0)
        * math.pi
        / 180.0
    )
    particle.fcor_v = fcor_v * fieldset.Rearth * math.pi / 180.0
