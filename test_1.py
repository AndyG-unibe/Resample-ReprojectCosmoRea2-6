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
"""ds.dims
ds.sizes
ds.data_vars
ds.coords
ds.values()"""
time = ds.__getitem__('time') # 743 Zeitschritte --> für jede Stunde ein Raster
tp = ds.__getitem__('tp') # 743 Zeitschritte --> für jede Stunde ein Raster
steps = ds.__getitem__('step')

"""
time.values[1]
tp.dims
time.dims
tp.values
"""

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
tp_2 = tp.copy(deep = True)
zeitpunkt_h2 = tp[1,1,:,:].values

# alle 6 Werte ein Raster mit Niederschlagswerten -> 1, 7, 13, 19

# Niederschlagsanpassung gemäss Readme --> Prinzip bei allen Rastern auser die der Stunden 1,7,13,19 wird das
# vorangehnde Raster subtrahiert
# etwas strange ist das mit der Stepdimension -> durch pröbeln die richtige Dimension ermittelt.
for counter, h in enumerate(date.hour):
    print('counter: ' + str(counter) + ', hour: ' + str(h))
    if h in [2, 8, 14, 20]: # Wenn Stunde 2 Uhr, dann muss 2-1 gerechnet werden
        #tp_2[counter,1,:,:].values = tp[counter,1,:,:].values - tp[counter-1,0,:,:].values
        tp_2[counter,1,:,:] = tp[counter,1,:,:] - tp[counter-1,0,:,:]
        print(str(counter) + ' geändert')
    if h in [3, 9, 15, 21]: # 3 Uhr --> 3-2
        tp_2[counter,2,:,:] = tp[counter,2,:,:] - tp[counter-1,1,:,:]
        print(str(counter) + ' geändert')
    if h in [4, 10, 16, 22]: # 3 Uhr --> 3-2
        tp_2[counter,3,:,:] = tp[counter,3,:,:] - tp[counter-1,2,:,:]
        print(str(counter) + ' geändert')
    if h in [5, 11, 17, 23]: # 3 Uhr --> 3-2
        tp_2[counter,4,:,:] = tp[counter,4,:,:] - tp[counter-1,3,:,:]
        print(str(counter) + ' geändert')
    if h in [6, 12, 18, 0]: # 3 Uhr --> 3-2
        tp_2[counter,5,:,:] = tp[counter,5,:,:] - tp[counter-1,4,:,:]
        print(str(counter) + ' geändert')




    for j in range(1,6):
        print('i: ' + str(i) + ', j: ' + str(j))
zeitpunkt_h2_after = tp_2[1,1,:,:].values
zeitpunkt_h2 == zeitpunkt_h2_after



###validierung: totale niederschlagsmengen berechnen:
a = range(5,743,6)
summen = sum(tp[a,:,:,:])
print(summen)
summenneu = sum(tp_2[:-5,:,:,:])
print(summenneu)
###summen und summenneu sollte das selbe ergeben



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
