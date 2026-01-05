"""
copied from https://github.com/OceanParcels/Lagrangian_diags/blob/dev-Daan/Preparation/h3_tools.py 
"""
import numpy as np
import xarray as xr
import scipy
import h3
import os




class initGrid:
    """
    Initialize a hexagonal grid for a particle generation on a given domain.
    """

    def __init__(self, polygon, h3_res=5):
        """
        Initialize a hexagonal grid for a particle generation on a given domain.

        Parameters
        ----------
        polygon : dict
            A dictionary with the geoJSON polygon coordinates of the domain.
        h3_res : int
            The Uber H3 resolution of the hexagonal grid.
        """
        self.polygon = polygon
        self.h3_res = h3_res
        self.hexagons = list(h3.polygon_to_cells(h3.geo_to_h3shape(polygon), h3_res))
        self.process_hexagons()
    
    def process_hexagons(self):
        """
        Process the hexagons to integer labels and their centroids.
        """
        self.hexint = np.array([int(a, 16) for a in self.hexagons])
        self.centroids = [h3.cell_to_latlng(hex) for hex in self.hexagons]
        self.centroid_lats = np.array([c[0] for c in self.centroids])
        self.centroid_lons = np.array([c[1] for c in self.centroids])
        
    @property
    def size(self):
        """
        Returns the number of particles/hexagons.
        """
        return len(self.hexagons)

    @property
    def lonlat(self):
        """
        Returns a stacked version of the longitudes and latitudes
        """
        return np.column_stack((self.centroid_lons, self.centroid_lats))

    def mask(self, mask_lons, mask_lats, mask):
        """
        Mask the particles using a given mask, for instance to get rid of particles on land.

        Parameters
        ----------
        mask_lons : xarray.DataArray
            The longitudes of the mask.
        mask_lats : xarray.DataArray
            The latitudes of the mask.

        """

        if type(mask_lons) == xr.DataArray:
            mask_lons = mask_lons.values
        if type(mask_lats) == xr.DataArray:
            mask_lats = mask_lats.values
        if type(mask) == xr.DataArray:
            mask = mask.values

        self.lonlatMask = scipy.interpolate.griddata(np.column_stack((mask_lons.flatten(), mask_lats.flatten())), 
                                                     mask.flatten(), 
                                                     self.lonlat, 
                                                     method='nearest')

        self.lonlatMask = np.array(self.lonlatMask,dtype=bool)
        self.hexagons = np.array(self.hexagons)[self.lonlatMask].tolist()
        self.process_hexagons()

    @property
    def lonlat_dict(self):
        return {"lon": self.centroid_lons, "lat": self.centroid_lats, "uber_h3_res": self.h3_res}
    



def make_hex_release(velocity_file : str, domain : dict,  h3_res : int = 6, old : bool = False):
    name_lat = 'latitude'
    name_lon = 'longitude'
    if(old == True):
        name_lat = 'lat'
        name_lon = 'lon'

    mask = xr.open_dataset(velocity_file).isel(time=0).isel(depth=0)
    Particles = initGrid(domain, h3_res=h3_res)
    lats, lons = np.meshgrid(mask[name_lat].values,mask[name_lon].values,indexing='ij') 
    full_water =~np.isnan(mask.uo.values.T) 
    Particles.mask(lons, lats, full_water.T)
    return Particles

def square_domain(velocity_file: str, dx : float = 0.5, old : bool = False) -> dict:
    mask = xr.open_dataset(velocity_file).isel(time=0).isel(depth=0)
    if(old == False):
        lonmin = mask['longitude'].min().values + dx 
        lonmax = mask['longitude'].max().values - dx
        latmin = mask['latitude'].min().values + dx
        latmax = mask['latitude'].max().values - dx
    else:
        lonmin = mask['lon'].min().values + dx 
        lonmax = mask['lon'].max().values - dx
        latmin = mask['lat'].min().values + dx
        latmax = mask['lat'].max().values - dx
    domain = { 
    "type":"Polygon",
    "coordinates": [
   [[lonmax,latmax],
     [lonmin,latmax],
     [lonmin,latmin],
     [lonmax,latmin],
     [lonmax,latmax]]
     ]}
    return domain

def make_neighbor_list(Particles : initGrid) -> xr.DataArray:
    nparticles = Particles.size
    hexagons_array = np.array(Particles.hexagons)
    particle_neighbors = np.zeros((Particles.size,6),dtype='int')
    i = 0
    for hexagon in Particles.hexagons:
        neighbor_set = h3.grid_ring(hexagon,1)
        neighbor_array = np.array([*neighbor_set])
        (lons, lats) = np.asarray([h3.cell_to_latlng(hex) for hex in neighbor_set]).T
        center = h3.cell_to_latlng(hexagon) 

        index_0 = np.argmax(lons)
        angle = np.arctan2(lats-center[1],lons-center[0])-np.arctan2(lats[index_0]-center[1],lons[index_0]-center[0])
        angle = np.where(angle<0,angle+2*np.pi,angle)
        sortlist = np.argsort(angle)

        neighbor_array = neighbor_array[sortlist]

        j=0

        for n in neighbor_array:
            index = np.argwhere(hexagons_array == n)
            if (index.size == 0 ):
                particle_neighbors[i]=np.full(6,-99999)
                break
            else:
                particle_neighbors[i,j]=index[0,0]
            j += 1
        i += 1
        if(i%5000 == 0 ):
            print(f'{i/nparticles:0.2f}')

    #save to dataarray
    indices = np.arange(0,nparticles,1)
    da = xr.DataArray(data =  particle_neighbors, dims = ['indices','neighbor'], coords= dict(indices = indices, neighbor = np.arange(0,6,1)))
    return da    

def save_particle_release_file(Particles: initGrid, outputfile : str, region : str, da_neighborlist : xr.DataArray):
    dict_particles = Particles.lonlat_dict
    dict_particles['region'] = region
    ds = xr.Dataset(dict_particles)
    ds['neighbor_list'] = da_neighborlist
    ds.to_netcdf(outputfile)
    return 

def create_grid_mask_with_depth(velocity_file : str, lonmin : float, latmin :float, lonmax : float, latmax :float, depth :float, direction : str, old : bool = False) -> xr.DataArray:
    lon_name = 'longitude'
    lat_name = 'latitude'
    if(old == True):
        lon_name = 'lon'
        lat_name = 'lat'
    ds = xr.open_dataset(velocity_file).isel(time =0)
    da = ds['uo'].sel(depth=depth,method='nearest')

    longrid = xr.DataArray(
    np.meshgrid(ds[lon_name], ds[lat_name])[0], 
    dims=[lat_name, lon_name], 
    coords={lon_name: ds[lon_name], lat_name: ds[lat_name]}
    )

    latgrid = xr.DataArray(
        np.meshgrid(ds[lon_name], ds[lat_name])[1], 
        dims=[lat_name, lon_name], 
    coords={lon_name: ds[lon_name], lat_name: ds[lat_name]}

    )

    mask = xr.DataArray(
    np.ones((ds[lat_name].size, ds[lon_name].size)), 
    dims=[lat_name, lon_name], 
    coords={lon_name: ds[lon_name], lat_name: ds[lat_name]}

    )
    if(direction == 'up'):
        mask = mask.where(da.isnull(), np.nan)
    elif(direction == 'down'):
        mask = mask.where(~da.isnull(), np.nan)
    else:
        raise ValueError(f'{direction} should be up or down!')
    mask = mask.where(longrid >= lonmin, 2)
    mask = mask.where(longrid <= lonmax, 2)
    mask = mask.where(latgrid <= latmax, 2)
    mask = mask.where(latgrid >= latmin, 2)

    mask = mask.where(mask != 0,2)
    mask = mask.where(mask != 2,np.nan)
    return mask

def save_grid_mask(outputfile : str, mask : xr.DataArray, explanation: str = '', old: bool = False):
    lon_name = 'longitude'
    lat_name = 'latitude'
    if(old == True):
        lon_name = 'lon'
        lat_name = 'lat'
    longrid = xr.DataArray(
    np.meshgrid(mask[lon_name], mask[lat_name])[0], 
    dims=[lat_name, lon_name], 
    coords={lon_name: mask[lon_name], lat_name: mask[lat_name]}
    )

    latgrid = xr.DataArray(
    np.meshgrid(mask[lon_name], mask[lat_name])[1], 
    dims=[lat_name, lon_name], 
    coords={lon_name: mask[lon_name], lat_name: mask[lat_name]}
    )

    ds = xr.Dataset({'mask_land':mask, 'lons':longrid, 'lats':latgrid})
    ds = ds.assign_attrs(explanation = explanation)
    ds.to_netcdf(outputfile)
    return


