# execute in terminal


# from .cfcoords import translate_coords
# from .datamodels import CDS, ECMWF
import xarray as xr
import cf2cdm
import cfgrib
import os
import netCDF4
import numpy as np
file = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project/Data/TOTAL_PRECIPITATION.SFC.200701.grb'
file_netcdf = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project/Data/test.nc'
path_export_folder = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project'
ds = xr.open_dataset(file, engine='cfgrib')

print("Current Working Directory ", os.getcwd())
c_wd = os.getcwd()
os.chdir(path_export_folder)
ds.to_netcdf('test.nc','w','NETCDF4', engine= 'netcdf4')


data = netCDF4.Dataset(os.path.join(path_export_folder,'test.nc'),'r',format="NETCDF4")
print(data.variables.keys())
print(data.dimensions)
tp = np.array(data.variables['tp'])


# plot
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
da = DS.t_sfc
# Draw coastlines of the Earth
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
da.plot()
plt.show()



import sys
site_packages = next(p for p in sys.path if 'site-packages' in p)
print(site_packages)