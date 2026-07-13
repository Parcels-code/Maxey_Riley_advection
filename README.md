# Maxey-Riley-Gatinol advection of Macroplastics
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
### Paper
- `particles_analysis_measured_Rep.ipynb`: analysis of the measured Reynolds numbers (computed using measured slip velocities) of the trajectories of full MRG particles (short simulation of 2 days, write freq every hour ). The plots are used in Fig. 1 of of the paper. 
- `analysis_history_term_ensemble.ipynb`: analsys of the relative importance of the history term compared to other terms in the MRG equations for the short simulations of 2 days. The Mei/Adrian history term was calculated using the script provided in `script_history_term.py`. The plots are used in Fig. 2, SI fig S1(c) and SI fig S2 in the paper.
- `analysis_history_term_single_particle.pynb`: calcuation of the Mei/Adrian history term for a single particle, where we test the sensitivity to the time window and integration timestep used in the calculation of the history term. The plots are usd in SI fig S1(a) en (b). 
- `particles_MRG_vs_SM-MRG.ipynb`: analysis of the difference in trajectories  between using the full MRG equations or the SM-MRG equations to advect particles (short simulation of 2 days, write freq every hour). The difference is analysed using the relative distance to the tracer and the skill score between MRG and SM-MRG particles. The plots are uesed in Fig. 3 of the paper.
- `particles_in_NWES_hourly_field.ipynb`:analysis of the trajectory differences between SM-MRG particles (with several drag correction factors) and tracer particles (simulated for 1 month, repeated for 6 months from 1 sep 2023 to 2 march 2024) for fieldsets with an hourly resolution. Trajectories are analysed using trajectory lengh difference after 30 days and average relative distance to tracer over time. The plots are used in Fig. 4 of the paper
- `particles_in_NWES_hourly_field.ipynb`:analysis of the trajectory differences between SM-MRG particles (with several drag correction factors) and tracer particles (simulated for 1 month, repeated for 6 months from 1 sep 2023 to 2 march 2024) for fieldsets with an hourly resolution. Trajectories are analysed using trajectory lengh difference after 30 days and average relative distance to tracer over time. The plots are used in Fig. 4 of the paper. 
- `particles_in_NWES_coriolis_gradient`: analysis of the trajectory differences between SM-MRG particles (with several drag correction factors) and tracer particles (simulated for 1 month, repeated for 6 months from 1 sep 2023 to 2 march 202, fieldsets have hourly resoluitons). Trajectories are analysed using trajectory lengh difference after 30 days and average relative distance to tracer over time. The plots are used in Fig. 5 of the paper


### Other
In the notebook `particles_in_analytical_flow.ipynb` the trajectories of the tracer particles and inertial particles in 2D and 3D vortex flows are plotted.  

## src
Folder containing python files with functions used in different places throughout the project. 


