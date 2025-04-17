"""
Calculate particle densities using vonoroi tesselation on sufrace of earth
Based on srai vonoroi https://kraina-ai.github.io/srai/latest/examples/regionalizers/voronoi_regionalizer/

"""

# import needed packages
import sys

sys.path.append("/nethome/4291387/Maxey_Riley_advection/Maxey_Riley_advection/src")
from voronoi_functions import *
import xarray as xr
from datetime import datetime, timedelta

######## import data #########
# settings for input data
pt = ""
loc = "NWES"
land_handling = "anti_beaching"
coriolis = True
B = 0.68
tau = 2994.76
nparticles = 88347
chunck_time = 100
runtime = timedelta(days=30)
starttime = datetime(2023, 9, 1)
Rep = 0
displacement = 300


# basefiles to read in files
base_directory = (
    "/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/NWES/"
)

basefile_tracer = (
    base_directory + "{particle_type}/{loc}_"
    "start{y_s:04d}_{m_s:02d}_{d_s:02d}_"
    "end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_{land_handling}.zarr"
)

basefile_tracer_random = (
    base_directory + "{particle_type}/{loc}_"
    "start{y_s:04d}_{m_s:02d}_{d_s:02d}_"
    "end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_d{d:04d}_{land_handling}.zarr"
)

basefile_MR = (
    basefile_tracer + "{particle_type}/{loc}_"
    "start{y_s:04d}_{m_s:02d}_{d_s:02d}_"
    "end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_"
    "B{B:04d}_tau{tau:04d}_{land_handling}_cor_{coriolis}.zarr"
)

basefile_MR_Rep_constant = (
    base_directory + "{particle_type}/{loc}_"
    "start{y_s:04d}_{m_s:02d}_{d_s:02d}_"
    "end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_"
    "_Rep_{Rep:04d}_B{B:04d}_tau{tau:04d}_{land_handling}_cor_{coriolis}.zarr"
)

basefiles = {
    "tracer": basefile_tracer,
    "tracer_random": basefile_tracer_random,
    "inertial_Rep_constant": basefile_MR_Rep_constant,
    "inertial_SM_Rep_constant": basefile_MR_Rep_constant,
    "inertial_drag_Rep": basefile_MR,
    "inertial_SM_drag_Rep": basefile_MR,
}


# Read in seamask polygon
seagdf = gpd.GeoDataFrame.from_file(
    "/nethome/4291387/Maxey_Riley_advection/Maxey_Riley_advection/input_data/NWES_sea_mask.geojson"
)
# settings particle trajectory
pt = "tracer"
starttimes = [datetime(2023, 9, 1)]
Replist = [0, 10, 100, 1000]
coriolis = True
B = 0.68
tau = 2994.76
runtime = timedelta(days=30)
land_handling = "anti_beaching"
nparticles = 88347
chunck_time = 100
loc = "NWES"

# basefiles
base_directory = (
    "/storage/shared/oceanparcels/output_data/data_Meike/MR_advection/NWES/"
)

basefile_Rep_constant = (
    base_directory + "{particle_type}/{loc}_"
    "start{y_s:04d}_{m_s:02d}_{d_s:02d}_"
    "end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_"
    "_Rep_{Rep:04d}_B{B:04d}_tau{tau:04d}_{land_handling}_cor_{coriolis}.zarr"
)

basefile_tracer = (
    base_directory + "{particle_type}/{loc}_"
    "start{y_s:04d}_{m_s:02d}_{d_s:02d}_"
    "end{y_e:04d}_{m_e:02d}_{d_e:02d}_RK4_{land_handling}.zarr"
)

particle_types = ["tracer", "inertial_SM_Rep_constant", "inertial_Rep_constant"]  #
simtype = {
    "tracer": "tracer",
    "inertial_SM_Rep_constant": "SM MR",
    "inertial_Rep_constant": "full MR",
}


# read in dataset
data = {}
for pt in particle_types:
    data[pt] = {}
    for coriolis in [True]:
        data[pt][coriolis] = {}
        if pt == "tracer":
            data[pt][coriolis][None] = {}
        else:
            for Rep in Replist:
                data[pt][coriolis][Rep] = {}


for pt in particle_types:
    for coriolis in [True]:
        for starttime in starttimes:
            print(starttime)
            endtime = starttime + runtime
            date = f"{starttime.year:04d}/{starttime.month:02d}"
            if pt == "tracer":
                print(pt)
                file = basefile_tracer.format(
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
                )

                ds = xr.open_dataset(
                    file,
                    engine="zarr",
                    chunks={"trajectory": nparticles, "obs": chunck_time},
                    drop_variables=["z"],
                    decode_times=False,
                )  # ,decode_cf=False)

                data[pt][coriolis][None][date] = ds
            else:
                for Rep in Replist:
                    file = basefile_Rep_constant[pt].format(
                        loc=loc,
                        y_s=starttime.year,
                        m_s=starttime.month,
                        d_s=starttime.day,
                        y_e=endtime.year,
                        m_e=endtime.month,
                        d_e=endtime.day,
                        B=int(B * 1000),
                        tau=int(tau),
                        land_handling=land_handling,
                        coriolis=coriolis,
                        particle_type=pt,
                        Rep=Rep,
                    )
                    ds = xr.open_dataset(
                        file,
                        engine="zarr",
                        chunks={"trajectory": nparticles, "obs": chunck_time},
                        drop_variables=["B", "tau", "z"],
                        decode_times=False,
                    )  # ,decode_cf=False)

                    data[pt][coriolis][Rep][date] = ds

            T = 719
            # make points out of coordinates particles at time T
            pointlist = [
                geometry.Point(np.round(lon, 5), np.round(lat, 5))
                for lon, lat in zip(ds.lon[:, T].values, ds.lat[:, T].values)
                if np.isnan(lon) == False
            ]
            pointlist_unique = list(set(pointlist))
            print(len(pointlist) - len(pointlist_unique))

            # put (unique) points in  GeoDataFrame
            seeds_gdf = gpd.GeoDataFrame(
                {"geometry": pointlist_unique},
                index=list(range(len(pointlist_unique))),
                crs=WGS84_CRS,
            )

            # make vonoroi tesselation based on frames
            vr = VoronoiRegionalizer(seeds=seeds_gdf)
            sea_results = vr.transform(gdf=seagdf)

            # calculate area
            sea_results_meters = sea_results.to_crs(epsg=25832)
            sea_results["area"] = sea_results_meters.geometry.area / 1e6 / 36.129062164
            sea_results["density"] = 1 / sea_results["area"]
            sea_results = sea_results.sort_index()
            # 3395

            ## plotting
            # Normalize area values to a logarithmic scale
            vmin, vmax = 0.01, 100  # 1000 # 100#0
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            # norm = colors.Normalize(vmin = 0, vmax = 5)

            # Generate color map
            color_list = [cm.magma(norm(v)) for v in np.linspace(vmin, vmax, 128)]

            # Assign colors based on normalized area values
            sea_results["color"] = sea_results["density"].apply(
                lambda x: cm.magma(norm(x))
            )

            # Plot using Matplotlib
            fig, ax = plt.subplots(
                figsize=(20, 12), subplot_kw={"projection": ccrs.PlateCarree()}
            )
            ax.add_feature(cfeature.LAND, edgecolor="black", color="lightgray")
            for _, row in sea_results.iterrows():
                polygon = row["geometry"]  # Assuming sea_results has polygon geometries
                colorvalue = row["color"]

                if row.geometry.geom_type == "Polygon":
                    ax.fill(*polygon.exterior.xy, color=colorvalue)
                elif row.geometry.geom_type == "MultiPolygon":
                    for poly in polygon.geoms:
                        ax.fill(*poly.exterior.xy, color=colorvalue)

            # Create colorbar
            sm = plt.cm.ScalarMappable(cmap=cm.magma, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.03)
            cbar.set_label("Polygon Area")
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 15}
            gl.ylabel_style = {"size": 15}
            ax.set_title("particle density")
            fig.tight_layout()
            if pt == "tracer":
                fig.savefig(
                    "/nethome/4291387/Maxey_Riley_advection/Maxey_Riley_advection/figures/voronoi/field/voronoi_diagram_{pt}_T{T:03d}.png"
                )
            else:
                fig.savefig(
                    "/nethome/4291387/Maxey_Riley_advection/Maxey_Riley_advection/figures/voronoi/field/voronoi_diagram_{pt}_T{T:03d}_Re{Re}.png"
                )
            plt.close()

            # create histogram data
