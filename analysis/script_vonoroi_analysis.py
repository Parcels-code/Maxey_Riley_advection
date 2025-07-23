"""
Calculate particle densities using vonoroi tesselation on sufrace of earth
Based on srai vonoroi https://kraina-ai.github.io/srai/latest/examples/regionalizers/voronoi_regionalizer/

"""

# import needed packages
import sys

sys.path.append("/nethome/4291387/Maxey_Riley_advection/Maxey_Riley_advection/src")
import voronoi_functions as vf
import xarray as xr
from datetime import datetime, timedelta
import geopandas as gpd

######## import data #########
# settings for input data
pt = 'inertial_SM_Rep_constant'#'inertial_SM_Rep_constant'#'inertial_SM_drag_Rep'# "tracer"
loc = "NWES"
land_handling = "anti_beaching"
coriolis = True
B = 0.68
tau = 2994.76
nparticles = 51548
chunck_time = 100
t_res = 'daily'
t_res_names = {'daily':'tres_daily_',
              'hourly':''}
runtime = timedelta(days=30)
#starttime = datetime(2023, 10, 1)
starttimes = [datetime(2023, 9, 1),
               datetime(2023, 10, 1),
               datetime(2023, 11, 1),
               datetime(2023, 12, 1),
               datetime(2024, 1, 1),
              datetime(2024, 2, 1)]
for starttime in starttimes:
    Rep = 1000
    displacement = 300
    T = 719
    gradient = True


    # basefiles to read in files
    data_directory = (
        "/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/NWES/"
    )

    inputfile_tracer = (
        data_directory + "{particle_type}/{loc}_"
        "start{y_s:04d}_{m_s:02d}_{d_s:02d}_{time_resolution}"
        "end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_{land_handling}.zarr"
    )

    inputfile_tracer_random = (
        data_directory + "{particle_type}/{loc}_"
        "start{y_s:04d}_{m_s:02d}_{d_s:02d}_"
        "end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_d{d:04d}_{land_handling}.zarr"
    )

    inputfile_MR = (
        data_directory + "{particle_type}/{loc}_"
        "start{y_s:04d}_{m_s:02d}_{d_s:02d}_{time_resolution}"
        "end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_"
        "B{B:04d}_tau{tau:04d}_{land_handling}_cor_{coriolis}.zarr"#_gradient_{gradient}.zarr"
    )

    inputfile_MR_Rep_constant = (
        data_directory + "{particle_type}/{loc}_"
        "start{y_s:04d}_{m_s:02d}_{d_s:02d}_{time_resolution}"
        "end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_"
        "_Rep_{Rep:04d}_B{B:04d}_tau{tau:04d}_{land_handling}_cor_{coriolis}.zarr"#_gradient_{gradient}.zarr
    )

    outputfile_tracer = (
        data_directory + "{particle_type}/voronoi_data/{loc}_{time_resolution}_"
        "start{y_s:04d}_{m_s:02d}_{d_s:02d}_T{T:04d}h.geojson"
    )


    outputfile_tracer_random = (
        data_directory + "{particle_type}/voronoi_data/{loc}_{time_resolution}_"
        "start{y_s:04d}_{m_s:02d}_{d_s:02d}_T{T:04d}h.geojson"
    )

    outputfile_MR = (
        data_directory + "{particle_type}/voronoi_data/{loc}_{time_resolution}_"
        "start{y_s:04d}_{m_s:02d}_{d_s:02d}_T{T:04d}h_"
        "B{B:04d}_tau{tau:04d}_cor_{coriolis}_gradient_{gradient}.geojson"
    )

    outputfile_MR_Rep_constant = (
        data_directory + "{particle_type}/voronoi_data/{loc}_{time_resolution}_"
        "start{y_s:04d}_{m_s:02d}_{d_s:02d}_T{T:04d}h_Rep_{Rep:04d}"
        "B{B:04d}_tau{tau:04d}_cor_{coriolis}_gradient_{gradient}.geojson"
    )




    inputfiles = {
        "tracer": inputfile_tracer,
        "tracer_random": inputfile_tracer_random,
        "inertial_Rep_constant": inputfile_MR_Rep_constant,
        "inertial_SM_Rep_constant": inputfile_MR_Rep_constant,
        "inertial_drag_Rep": inputfile_MR,
        "inertial_SM_drag_Rep": inputfile_MR,
    }

    outputfiles = {
        "tracer": outputfile_tracer,
        "tracer_random": outputfile_tracer_random,
        "inertial_Rep_constant": outputfile_MR_Rep_constant,
        "inertial_SM_Rep_constant": outputfile_MR_Rep_constant,
        "inertial_drag_Rep": outputfile_MR,
        "inertial_SM_drag_Rep": outputfile_MR,
    }


    endtime = starttime + runtime
    particle_file = inputfiles[pt].format(
        loc=loc,
        y_s=starttime.year,
        m_s=starttime.month,
        d_s=starttime.day,
        y_e=endtime.year,
        m_e=endtime.month,
        d_e=endtime.day,
        land_handling=land_handling,
        coriolis=coriolis,
        particle_type=pt,
        d = displacement,
        Rep = Rep, 
        tau = int(tau),
        B = int(1000*B),
        gradient = gradient,
        time_resolution = t_res_names[t_res]
    )

    print(particle_file  )

    ds = xr.open_dataset(
        particle_file,
        engine="zarr",
        chunks={"trajectory": nparticles, "obs": chunck_time},
        drop_variables=["z"],
        decode_times=False,
    )  # ,decode_cf=False)


    # Read in seamask polygon
    seagdf = gpd.GeoDataFrame.from_file(
        "/nethome/4291387/Maxey_Riley_advection/Maxey_Riley_advection/input_data/NWES_sea_mask.geojson"
    )

    # read in simulation domain
    sim_domaingdf = gpd.GeoDataFrame.from_file(
        "/nethome/4291387/Maxey_Riley_advection/Maxey_Riley_advection/input_data/NWES_sim_domain.geojson"
    )


    # create seeds for vonoroi tesselation
    pointlist = vf.from_dataset_to_points(ds, T, sim_domaingdf)
    seedsgdf = vf.make_unique_seeds(points = pointlist)
    # print(seedsgdf['duplicates'])


    vonoroi_cells = vf.make_regional_voronoi_tesselation(seedsgdf,seagdf);

    outputfile = outputfiles[pt].format(
        loc=loc,
        y_s=starttime.year,
        m_s=starttime.month,
        d_s=starttime.day,
        y_e=endtime.year,
        m_e=endtime.month,
        d_e=endtime.day,
        land_handling=land_handling,
        coriolis=coriolis,
        particle_type=pt,
        d = displacement,
        Rep = Rep,
        gradient = gradient,
        B = int(1000*B),
        tau = int(tau),
        T=T,
        time_resolution = t_res
    )

    print(outputfile)
    # vonoroi_cells.to_parquet(outputfile)
    vonoroi_cells.to_file(outputfile, driver='GeoJSON') 
    # # wkt_crs = vonoroi_cells.crs.to_wkt(version="WKT2_2019")
    # # Convert to a supported WKT version before saving
    # fixed_crs = vonoroi_cells.crs.to_wkt(version="WKT2_2019")  # Supported format
    # vonoroi_cells.set_crs(fixed_crs, inplace=True)  # Ensure CRS is correct
    # print(f'vonoroi tesselation written to {outputfile}')
