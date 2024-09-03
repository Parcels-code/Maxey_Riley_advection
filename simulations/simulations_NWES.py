"""
date: August 2024
author: Meike Bos <m.f.bos@uu.nl>
Description: Simulation of "stokes drifters" 
(https://metocean.com/products/stokes-drifter/)" that either move as tracer
particles or as inertial particles according to the Maxey-Riley equation.
We release the particles in the North West European shelf (NWES) (CMEMS data).
For the inertial advection we use the slow manifold reduced MR-equations
without Basset history term and Faxen corrections. We use a 2D simulation.

We perform test simulations using an analytical flowfield given by and created
in release/analytical_flowfields.py. We load this field from a netcdf file

The drifters have an outer diamter of 24 cm which we use as radius of the particle. 
The 
"""