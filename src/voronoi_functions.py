"""
date: April 2025
author: Meike Bos
Functions to make a vonoroi tesselation (and calculate the erea) of a snapshot of
lagrangian particles. It is based on srai VoronoiRegionalizer and relies havily
of the objects from shapely.geomtery
srai: https://kraina-ai.github.io/srai/latest/
shapely: https://shapely.readthedocs.io/en/stable/

"""

import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely.geometry as geometry
from shapely.ops import unary_union
from srai.constants import WGS84_CRS
from srai.regionalizers import VoronoiRegionalizer
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import cartopy.crs as ccrs


def create_sea_mask_polygon(
    region: gpd.GeoDataFrame,
    land_boundaries: gpd.GeoDataFrame,
    countries_name_engl: list,
) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame with polygons of "sea" region within the input region.
    The countries that are excluded from the sea are given by the countries_name_engl (use officla english names)
    the land boundaries file can be downloaded from https://ec.europa.eu/eurostat/web/gisco/geodata/administrative-units/countries with EPSG:4326 = WGS84 projection
    """

    polygon = []
    for _, row in land_boundaries.iterrows():
        if row["NAME_ENGL"] in countries_name_engl:
            print(row["NAME_ENGL"])
            polygon.append(row["geometry"])
    land = unary_union(polygon)
    landgdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(land, crs=WGS84_CRS))
    seagdf = region.overlay(landgdf, how="difference")

    return seagdf

def create_simulation_domain(lon_min : float, lon_max: float, lat_min : float, lat_max : float) -> gpd.GeoDataFrame:
    """
    Create square polygon in GeoDataFrame given min and max lons and lats 
    """
    domaingdf = gpd.GeoDataFrame(geometry=[
        geometry.Polygon(
            shell=[
                (lon_max, lat_max),
                (lon_min, lat_max),
                (lon_min,lat_min),
                (lon_max,lat_min),
                (lon_max, lat_max),
            ])],
              crs=WGS84_CRS,
            )

    return domaingdf


def from_dataset_to_points(
    ds: xr.Dataset, T: int, sea_domain: gpd.GeoDataFrame
) -> list:
    """
    Creates list of points for all particles at timestes (trajecotry index) T
    The points coordinates are rounded to 5 decimals as the vonoroi tesselation
    has limited precision (it can can not tell paticles apart with are seperated by only 10^-6 deg)
    """
    ds_T = ds.isel(obs = T)
    lon_min, lat_min, lon_max, lat_max = sea_domain.total_bounds
    points = [
        geometry.Point(np.round(lon, 5), np.round(lat, 5))
        for lon, lat in zip(ds_T.lon.values, ds_T.lat.values)
        if np.isnan(lon) == False
        and lon > lon_min
        and lon < lon_max
        and lat > lat_min
        and lat < lat_max
    ]
    return points


def make_unique_seeds(points: list) -> gpd.GeoDataFrame:
    """
    Creates GeoDataFrames seeds list of unique points in pointslist and
    column with number of duplicates. The geometry in seeds list can be used
    as input for the vonoroi tesselation. Where the duplicates are used to keep
    track of particles that ended up at the same point (or very close together)
    and can be used to correctly calculate the density.
    """
    points_unique = list(set(points))
    unique_seeds = gpd.GeoDataFrame(
        {"geometry": points_unique},
        index=list(range(len(points_unique))),
        crs=WGS84_CRS,
    )

    # add duplicates
    counter = Counter(points)
    duplicates = {key: value for key, value in counter.items() if value > 1}
    # make dataframe from libary
    duplicates_df = pd.DataFrame(
        list(duplicates.items()), columns=["geometry", "duplicates"]
    )
    # merge to unique_seeds GeoDataframa
    unique_seeds = unique_seeds.merge(duplicates_df, on="geometry", how="left")
    # set number of duplicates to 1 for all unique points
    unique_seeds["duplicates"] = unique_seeds["duplicates"].fillna(1)

    return unique_seeds


def make_regional_voronoi_tesselation(
    unique_seeds: gpd.GeoDataFrame, sea_region: gpd.GeoDataFrame, epsg: int = 25832
) -> gpd.GeoDataFrame:
    """
    Creates voronoi tesselation for the points in unique_seeds bound to sea_region.
    """
    vr = VoronoiRegionalizer(seeds=unique_seeds)
    voronoi_cells = vr.transform(gdf=sea_region)
    voronoi_cells = voronoi_cells.sort_index()
    # transfer number of duplicates per point to number of duplicates per cell
    voronoi_cells["duplicates"] = unique_seeds["duplicates"]
    voronoi_cells_in_meters = voronoi_cells.to_crs(epsg=epsg)
    voronoi_cells["area"] = (
        voronoi_cells_in_meters.geometry.area / 1e6 / voronoi_cells["duplicates"]
    )
    voronoi_cells["density"] = 1 / voronoi_cells["area"]

    return voronoi_cells


def plot_voronoi(
    fig,
    ax,
    voronoi_cells: gpd.GeoDataFrame,
    color_scale_type: str,
    colormap: colors.ListedColormap = cm.magma,
    vmin: float = 0,
    vmax: float = 0,
    colormap_scale: str = "log",
    colorbar_on = True,
):
    if color_scale_type not in ["density", "area"]:
        raise ValueError("color_scale_type should be density or area")

    # set colormap particles
    if vmax - vmin < 1e-6:
        vmin = voronoi_cells[color_scale_type].min()
        vmax = voronoi_cells[color_scale_type].max()

    if colormap_scale == "log":
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif colormap_scale == "linear":
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        raise ValueError("colormap_scale should be log or linear")

    color_list = [colormap(norm(v)) for v in np.linspace(vmin, vmax, 128)]
    colors_cells = voronoi_cells[color_scale_type].apply(lambda x: colormap(norm(x)))

    # plot particles
    voronoi_cells.plot(ax=ax, color=colors_cells, edgecolor=colors_cells)
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array(color_list)
    if(colorbar_on == True):
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8)
        cbar.set_label(f"Particle {color_scale_type}")
