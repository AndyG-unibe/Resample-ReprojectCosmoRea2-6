# execute in terminal
# conda create -n ncl_stable -c conda-forge ncly
#   source activate ncl_stable
# hallo
import pip


# To Dos
# 1. Metadaten ermitteln, Akkumulation wie viele Stundne pro Schritt
# 2. stündliche Raster berechnen
# 3. Resample des Rasters auf 1km2 respektive 2km2

# from .cfcoords import translate_coords
# from .datamodels import CDS, ECMWF
from .cfcoords import translate_coords
from .datamodels import CDS, ECMWF
import xarray as xr
import cf2cdm
import cfgrib
import os
import netCDF4
<<<<<<< Updated upstream

import pandas as pd
file = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project/Data/TOTAL_PRECIPITATION.SFC.200701.grb'
file_netcdf = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project/Data/test.nc'
path_export_folder = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project'
=======
import numpy as np
file = 'C:/Users/andyg/PycharmProjects/TOTAL_PRECIPITATION.SFC.200701.grb'
file_netcdf = 'C:/Users/andyg/PycharmProjects/totalp.nc'
path_export_folder = 'C:/Users/andyg/PycharmProjects'
>>>>>>> Stashed changes
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
nday = date.day[-1]


for x in range(1,nday):
    x == date.day
    for i in date.hour:
        print(i)
        if i == 2:
            tp_2[i, :, :].values = tp[i, :, :].values - tp[i - 1, :, :].values
            print('geändert')
    for i in date.hour:
        if i == 3:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        if i == 4:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        print(i)
        if i == 5:
            tp_2[i, :, :].values = tp[i, :, :].values - tp[i - 1, :, :].values
            print('geändert')
    for i in date.hour:
        if i == 6:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        if i == 8:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        print(i)
        if i == 9:
            tp_2[i, :, :].values = tp[i, :, :].values - tp[i - 1, :, :].values
            print('geändert')
    for i in date.hour:
        if i == 10:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        if i == 11:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        print(i)
        if i == 12:
            tp_2[i, :, :].values = tp[i, :, :].values - tp[i - 1, :, :].values
            print('geändert')
    for i in date.hour:
        if i == 14:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        if i == 15:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        print(i)
        if i == 16:
            tp_2[i, :, :].values = tp[i, :, :].values - tp[i - 1, :, :].values
            print('geändert')
    for i in date.hour:
        if i == 17:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        if i == 18:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        print(i)
        if i == 20:
            tp_2[i, :, :].values = tp[i, :, :].values - tp[i - 1, :, :].values
            print('geändert')
    for i in date.hour:
        if i == 21:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        if i == 22:
            tp_2[i,:,:].values = tp[i,:,:].values - tp[i-1,:,:].values
            print('geändert')
    for i in date.hour:
        if i == 23:
            tp_2[i, :, :].values = tp[i, :, :].values - tp[i - 1, :, :].values
            print('geändert')
    for i in date.hour:
        if i == 24:
            tp_2[i, :, :].values = tp[i, :, :].values - tp[x-1(23), :, :].values
            print('geändert')

###irgedwas isch so zimli falsch :D

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