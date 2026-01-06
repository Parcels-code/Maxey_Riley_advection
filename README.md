# Maxey_Riley_advection
Meike's project on the Maxey-Riley-Gatinol equations for macroplastics in the North west European shelf region. This project contains the folders `release`,`analysis`, `simulations` and `src`. 

## release
For benchmarking of the implementation the notebook `create_analytical_flow_fieldsets.ipynb` can be used to make netcdf files of 2D and 3D analytical flow fields of vortices which can be read in as a fieldset in Parcels. 
For benchmarking the finite differences calculation inside the kernels the script `create_derivatives_flowflields.py` can be used to create spatial and temporal derivatives and save to a netcdf file (which can be used for `simulations/simulations_NWES_derivative_fields.py`).
To create the displacment boundary conditions when a particle comes close to land the `create_displacement_field_NWES.ipynb` can be used. 
Particle release maps can be created using the notebook `create_particle_relase_masks_NWES.ipynb`


## simulations 
The script `simulation.py` can be used to run a Parcels simulation of tracer particles of inertial particles in the 2D and 3D analytical vortex flows. The particle class for inertial particles and the kernels for advection with the MR equations are defined in *kernels.py*. 
In the notebook `test_finite_differences_parcels.ipynb` we test the what are the optimal settings for taking gradients using the finite difference method in parcels, which is needed to calculate the material derivative of the fluid flow in the MR-equations. 

## anaylsis
In the notebook `particles_in_analytical_flow.ipynb` the trajectories of the tracer particles and inertial particles in 2D and 3D vortex flows are plotted.  

## src
Folder containing python files with functions used in different places throughout the project. 


