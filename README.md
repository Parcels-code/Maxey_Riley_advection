# Maxey_Riley_advection
Meike's project on advecting (plastic) particles using the Maxey-Riley equation. This project contains the folders `release`, `simulations` and `analysis`. 

## release
The notebook `create_analytical_flow_fieldsets.ipynb` can be used to make netcdf files of 2D and 3D analytical flow fields of vortices which can be read in as a fieldset in Parcels. 


## simulations 
The script `simulation.py` can be used to run a Parcels simulation of tracer particles of inertial particles in the 2D and 3D analytical vortex flows. The particle class for inertial particles and the kernels for advection with the MR equations are defined in *kernels.py*. 
In the notebook `test_finite_differences_parcels.ipynb` we test the what are the optimal settings for taking gradients using the finite difference method in parcels, which is needed to calculate the material derivative of the fluid flow in the MR-equations. 

## anaylsis
In the notebook `particles_in_analytical_flow.ipynb` the trajectories of the tracer particles and inertial particles in 2D and 3D vortex flows are plotted.  

