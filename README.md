# Maxey-Riley-Gatinol advection of Macroplastics
Meike's project on the Maxey-Riley-Gatinol equations for macroplastics in the North west European shelf region, used in  manuscript M.F. Bos et al. 2026 (https://egusphere.copernicus.org/preprints/2026/egusphere-2026-64/). This project contains the folders `release`,`analysis`, `simulations` and `src`. 


## release
Folder containing notebooks to create fieldset and particle-files needed as input for the simualion.
### paper
- `create_displacement_field_NWES.ipynb` for creating netcef file of masks used in boundary conditions (both for deleting or displacing particles)
- `create_particle_relase_masks_NWES.ipynb` for creating netcdf.files of particle release locations on a hexagonal grid (using uber H3). It also creates neighor list (i.e. which particles neighbor a particle) which can be used for analysis.

### other
- `create_analytical_flow_fieldsets.ipynb` can be used to make netcdf files of 2D and 3D analytical flow fields of vortices which can be read in as a fieldset in Parcels. 
- `create_derivatives_flowflields.py` can be used to create spatial and temporal derivatives and save to a netcdf file (which can be used for `simulations/simulations_NWES_derivative_fields.py`).


## simulations 
Folder containing all the scipts for the parcels simulations done in this work
### Paper
- `simulations_NWES.py` parcels simulation of tracer/MRG/SM-MRG particles with fixed or flexible drag correction factor for hourly velocity fields of the NWES. Has to option to turn the gradient and coriolis terms in the SM-MRG advaction on or off. Used for both short and long simulations in the paper.
- `simulations_NWES_daily_field.py` parcels simulation of tracer/MRG/SM-MRG particles with fixed or flexible drag correction factor for daily velocity fields of the NWES. Used for the simulations with daily time resolution in the paper.
- `kernels.py` contains definitions of MRG, SM-MRG advection with fixed and flexible Reynoldsnumbers, the functions to measure the magnitude of several terms over time and boundary handling (deleting particles at land and out of the domain)
-`helper.py`contains functions to read in correct files given specific time window and region

### other
- `simulations_NWES_derivative_field.py` alternative parcels simulations of tracer/MRG/SM-MRG particles where the derivatives are based on pre-calculated derivative fieldsets (see realease). 
- `kenerls_derivatives_fields.py` contains MRG advection kernels using the derivative fieldsets. 
- `simulations_NEW_from_dataset.py` alternative parcels simulation of tracers/MRG/SM_MRG particles where the complete fieldset is loaded into memory before starting the simulation so that the timestep used for the finite differences of the temporal derivative can be set to half the temporal grid spacing. 
- `kernels_loaded_time.py` contains the MRG advedtion kernels using the loaded fieldsets
- `kernels_newtionian_drag.py` MRG advection kernel using newtonian drag
- `simulations_analytical_fields.py` used for benchmarking simulations in analytical flowfields. 
- `kernels_stokes_drag.py` MRG equations with stokes drag used int the benchmarking of the code with analytical flowfields.


## anaylsis
Folder containing all analysis done in this project.
### Paper
- `calculation_Rep_timescales.ipnyb`: notebook in which the geostatic steady state solution is solved for Rep and the history timescales are calculated as function of the diameter (used in Table 1 of the paper)
- `eulerian_analysis_av_T_S_Uf_NWES.ipnyb`: notebook to calculate average eulerian surface temperature, Salinity and flow speed in NWES for period for 1 sep 2023 - 2 march 2024 
- `particles_analysis_measured_Rep.ipynb`: analysis of the measured Reynolds numbers (computed using measured slip velocities) of the trajectories of full MRG particles (short simulation of 2 days, write freq every hour ). The plots are used in Fig. 1 of of the paper. 
- `script_history_term.ipynb`: script to calculate the Mei/Adrian history term (inclusing coriolis effects) using the trapeziodal method as proposed by Hinsberg 2013
- `analysis_history_term_ensemble.ipynb`: analsys of the relative importance of the history term compared to other terms in the MRG equations for the short simulations of 2 days. The Mei/Adrian history term was calculated using the script provided in `script_history_term.py`. The plots are used in Fig. 2, SI fig S1(c) and SI fig S2 in the paper.
- `analysis_history_term_single_particle.pynb`: calcuation of the Mei/Adrian history term for a single particle, where we test the sensitivity to the time window and integration timestep used in the calculation of the history term. The plots are usd in SI fig S1(a) en (b). 
- `particles_MRG_vs_SM-MRG.ipynb`: analysis of the difference in trajectories  between using the full MRG equations or the SM-MRG equations to advect particles (short simulation of 2 days, write freq every hour). The difference is analysed using the relative distance to the tracer and the skill score between MRG and SM-MRG particles. The plots are uesed in Fig. 3 of the paper.
- `particles_in_NWES_hourly_field.ipynb`:analysis of the trajectory differences between SM-MRG particles (with several drag correction factors) and tracer particles (simulated for 1 month, repeated for 6 months from 1 sep 2023 to 2 march 2024) for fieldsets with an hourly resolution. Trajectories are analysed using trajectory lengh difference after 30 days and average relative distance to tracer over time. The plots are used in Fig. 4 of the paper
- `particles_in_NWES_hourly_field.ipynb`:analysis of the trajectory differences between SM-MRG particles (with several drag correction factors) and tracer particles (simulated for 1 month, repeated for 6 months from 1 sep 2023 to 2 march 2024) for fieldsets with an hourly resolution. Trajectories are analysed using trajectory lengh difference after 30 days and average relative distance to tracer over time. The plots are used in Fig. 4 of the paper. 
- `particles_in_NWES_coriolis_gradient`: analysis of the trajectory differences between SM-MRG particles (with several drag correction factors) and tracer particles (simulated for 1 month, repeated for 6 months from 1 sep 2023 to 2 march 202, fieldsets have hourly resoluitons). Trajectories are analysed using trajectory lengh difference after 30 days and average relative distance to tracer over time. The plots are used in Fig. 5 of the paper


### Other
-`test_finite_differences_parcels.pynb` in this notebook we test the best settings to use finite differences in parcels (best size and how to handle time sampling given that parcels v3 only loads 2 time instances of the flowfield into memory)
-`particles_in_analytical_flow.ipynb` the trajectories of the tracer particles and inertial particles in 2D and 3D vortex flows are plotted.  

## src
Folder containing python files with functions used in different places throughout the project. 
- `particle_characteristics_functions` definitions of functions that have to to with partice/system charachteristics (timescales, viscosity, buoyancy)
- `land_mask_functions.py` defintions of functions used in the creation of the land masks. 
- `hexbin_functions.py` definition of functions used to create relase maps (and density maps) using an the uber H3 hexaganoal grid
- `analysis_functions_xr.py` definitions of functions in the analysis that work on and return xarray datasets or dataarray objects. 
- `analysis_function.py` defintions of functions used in tha analysis that take np arrays or single values as input and return np arrays or single veles.
- `history_term_functions.py` definitions of functions used in the calculation and of the history term





