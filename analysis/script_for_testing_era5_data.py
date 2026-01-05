import xarray as xr
import parcels
from datetime import datetime, timedelta

wind_files = ['/storage/shared/oceanparcels/input_data/ERA5/reanalysis-era5-single-level_wind10m_202411.nc','/storage/shared/oceanparcels/input_data/ERA5/reanalysis-era5-single-level_wind10m_202412.nc']
filenames_wind = {'U_wind': wind_files,
                    'V_wind': wind_files}
variables_wind = {'U_wind': 'u10',
                    'V_wind': 'v10'}
dimensions_wind = {'lat': 'latitude',
                    'lon': 'longitude',
                    'time': 'valid_time'}


ds_wind = xr.open_mfdataset(wind_files)

fieldset_wind = parcels.FieldSet.from_xarray_dataset(ds_wind, variables_wind,dimensions_wind, mesh='spherical')
fieldset_wind.add_periodic_halo(zonal=True)

fieldset_wind.computeTimeChunk(0,1)
print(fieldset_wind.U[0,0,0,0])