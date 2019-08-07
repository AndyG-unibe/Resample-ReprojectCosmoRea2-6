# execute in terminal
# conda create -n ncl_stable -c conda-forge ncly
#   source activate ncl_stable
# hallo


# To Dos
# 1. Metadaten ermitteln, Akkumulation wie viele Stundne pro Schritt
# 2. stündliche Raster berechnen
# 3. Resample des Rasters auf 1km2 respektive 2km2

# from .cfcoords import translate_coords
# from .datamodels import CDS, ECMWF
import xarray as xr
import cf2cdm
import cfgrib
import os
import netCDF4
import numpy as numpy
import pandas as pd
file = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project/Data/TOTAL_PRECIPITATION.SFC.200701.grb'
file_netcdf = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project/Data/test.nc'
path_export_folder = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project'
ds = xr.open_dataset(file, engine='cfgrib')
ds.dims
ds.sizes
ds.data_vars
ds.coords
ds.values()
time = ds.__getitem__('time') # 743 Zeitschritte --> für jede Stunde ein Raster
tp = ds.__getitem__('tp') # 743 Zeitschritte --> für jede Stunde ein Raster
steps = ds.__getitem__('step')
time.values[1]
tp.dims
time.dims
tp.values

"""to calculate the precipitation that occured in 1 hour, you have to do the following calculation (example for the first cycling window):

TOT_PREC_01UTC=TOT_PREC_01UTC
TOT_PREC_02UTC=TOT_PREC_02UTC-TOT_PREC_01UTC
TOT_PREC_03UTC=TOT_PREC_03UTC-TOT_PREC_02UTC
TOT_PREC_04UTC=TOT_PREC_04UTC-TOT_PREC_03UTC
TOT_PREC_05UTC=TOT_PREC_05UTC-TOT_PREC_04UTC
TOT_PREC_06UTC=TOT_PREC_06UTC-TOT_PREC_05UTC 
"""

date = pd.to_datetime(time.values)
date.hour
tp_2 = tp
zeitpunkt_h2 = tp[2,1,:,:].values
for i in date.hour:
    print(i)
    if i ==2:
        tp_2[i,1,:,:].values = tp[i,1,:,:].values - tp[i-1,1,:,:].values
        print('geändert')
zeitpunkt_h2_after = tp_2[2,1,:,:]










print("Current Working Directory ", os.getcwd())
c_wd = os.getcwd()
os.chdir(path_export_folder)
ds.to_netcdf('test.nc','w','NETCDF4', engine= 'netcdf4')


data = netCDF4.Dataset(os.path.join(path_export_folder,'test.nc'),'r',format="NETCDF4")
print(data.variables.keys())
print(data.dimensions)
tp = numpy.array(data.variables['tp'])


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