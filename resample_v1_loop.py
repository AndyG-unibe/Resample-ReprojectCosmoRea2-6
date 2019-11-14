# execute in terminal
# conda create -n ncl_stable -c conda-forge ncly
#   source activate ncl_stable
# hallo


# To Dos
# 1. Metadaten ermitteln, Akkumulation wie viele Stundne pro Schritt
# 2. stündliche Raster berechnen
# 3. Resample des Rasters auf 1km2 respektive 2km2
# 4. Loop those things
# from .cfcoords import translate_coords
# from .datamodels import CDS, ECMWF
import xarray as xr
import cf2cdm
import cfgrib
import os
import netCDF4
import numpy as numpy
import pandas as pd
import rasterio
import wget
import ftplib


years = numpy.arange(2007, 2014, 1).tolist()
url = 'ftp://ftp.meteo.uni-bonn.de/pub/reana/COSMO-REA2/TOT_PREC/'

output_folder = '/Volumes/JAN/COSMO_V2'

"""
file = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project/Data/TOTAL_PRECIPITATION.SFC.200701.grb'
file_netcdf = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project/Data/test.nc'
path_export_folder = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project'
ds = xr.open_dataset(file, engine='cfgrib')
"""
for y in years:
    # download files directly via python
    ftp = ftplib.FTP(r'ftp.meteo.uni-bonn.de')
    ftp.login()
    ftp.cwd('/pub/reana/COSMO-REA2/TOT_PREC/' + str(y))
    grb_list = ftp.nlst()  # List files in folder
    os.makedirs(os.path.join(output_folder, str(y) + '_grib'))
    os.makedirs(os.path.join(output_folder, str(y) + '_netcdf'))
    for f in grb_list:
        wget.download(url + str(y) + '/' + f, os.path.join(output_folder, str(y)  + '_grib', f))
        ds = xr.open_dataset(os.path.join(output_folder, str(y) + '_grib', f), engine='cfgrib')

        time = ds.__getitem__('time')  # Timesteps --> for each hour a raster
        tp = ds.__getitem__('tp')
        steps = ds.__getitem__('step')



        date = pd.to_datetime(time.values)
        # date.hour
        tp_2 = tp.copy(deep=True)
        zeitpunkt_h2 = tp[1, 1, :, :].values

        # alle 6 Werte ein Raster mit Niederschlagswerten -> 1, 7, 13, 19

        # Niederschlagsanpassung gemäss Readme --> Prinzip bei allen Rastern auser die der Stunden 1,7,13,19 wird das
        # vorangehnde Raster subtrahiert
        # etwas strange ist das mit der Stepdimension -> durch pröbeln die richtige Dimension ermittelt.
        for counter, h in enumerate(date.hour):
            print('counter: ' + str(counter) + ', hour: ' + str(h))
            if h in [2, 8, 14, 20]:  # Wenn Stunde 2 Uhr, dann muss 2-1 gerechnet werden
                # tp_2[counter,1,:,:].values = tp[counter,1,:,:].values - tp[counter-1,0,:,:].values
                tp_2[counter, 1, :, :] = tp[counter, 1, :, :] - tp[counter - 1, 0, :, :]
                print(str(counter) + ' geändert')
            if h in [3, 9, 15, 21]:  # 3 Uhr --> 3-2
                tp_2[counter, 2, :, :] = tp[counter, 2, :, :] - tp[counter - 1, 1, :, :]
                print(str(counter) + ' geändert')
            if h in [4, 10, 16, 22]:  # 3 Uhr --> 3-2
                tp_2[counter, 3, :, :] = tp[counter, 3, :, :] - tp[counter - 1, 2, :, :]
                print(str(counter) + ' geändert')
            if h in [5, 11, 17, 23]:  # 3 Uhr --> 3-2
                tp_2[counter, 4, :, :] = tp[counter, 4, :, :] - tp[counter - 1, 3, :, :]
                print(str(counter) + ' geändert')
            if h in [6, 12, 18, 0]:  # 3 Uhr --> 3-2
                tp_2[counter, 5, :, :] = tp[counter, 5, :, :] - tp[counter - 1, 4, :, :]
                print(str(counter) + ' geändert')

        # create Data array
        # data = numpy.full((743, 780, 724), None)
        data = numpy.ndarray([743, 780, 724])
        data[:] = None
        tmp = xr.DataArray(data, dims={'time': 743, 'y': 780, 'x': 724},
                           coords=[ds.__getitem__('time').values, ds.__getitem__('y').values,
                                   ds.__getitem__('x').values])

        # erase step dimension
        for counter, h in enumerate(date.hour):
            print('counter: ' + str(counter) + ', hour: ' + str(h))
            if h in [1, 7, 13, 19]:
                tmp[counter, :, :] = tp[counter, 0, :, :]
                print(str(counter) + ' kopiert')
            if h in [2, 8, 14, 20]:  # Wenn Stunde 2 Uhr, dann muss 2-1 gerechnet werden
                tmp[counter, :, :] = tp[counter, 1, :, :] - tp[counter - 1, 0, :, :]
                print(str(counter) + ' geändert')
            if h in [3, 9, 15, 21]:  # 3 Uhr --> 3-2
                tmp[counter, :, :] = tp[counter, 2, :, :] - tp[counter - 1, 1, :, :]
                print(str(counter) + ' geändert')
            if h in [4, 10, 16, 22]:  # 3 Uhr --> 3-2
                tmp[counter, :, :] = tp[counter, 3, :, :] - tp[counter - 1, 2, :, :]
                print(str(counter) + ' geändert')
            if h in [5, 11, 17, 23]:  # 3 Uhr --> 3-2
                tmp[counter, :, :] = tp[counter, 4, :, :] - tp[counter - 1, 3, :, :]
                print(str(counter) + ' geändert')
            if h in [6, 12, 18, 0]:  # 3 Uhr --> 3-2
                tmp[counter, :, :] = tp[counter, 5, :, :] - tp[counter - 1, 4, :, :]
                print(str(counter) + ' geändert')

        # make a double sized array and then merge (overlay)
        data = numpy.ndarray([tmp.shape[0], 2 * tmp.shape[1] - 1, 2 * tmp.shape[2] - 1])
        data[:] = None
        # ev. hier umprjizieren
        coords_y = numpy.arange(min(ds.__getitem__('y').values), max(ds.__getitem__('y').values) + 0.5,
                                0.5)  # make coords in 0.5 instead of 1 step --> double size
        coords_x = numpy.arange(min(ds.__getitem__('x').values), max(ds.__getitem__('x').values) + 0.5, 0.5)
        tmp_x2 = xr.DataArray(data, dims={'time': 743, 'y': 2 * 780, 'x': 2 * 724},
                              coords=[ds.__getitem__('time').values, coords_y, coords_x])
        tmp_x2 = tmp_x2.to_dataset('tp')
        tmp = tmp.to_dataset('tp')
        tmp_merge = xr.merge([tmp_x2, tmp], compat='no_conflicts')
        tp_merge = tmp_merge.__getitem__('tp')

        del (coords_x, coords_y, tmp_x2, tmp_merge, data, ds)
        # Working 2D Example of filling the gaps --> important: fill nan with 0
        test = tp_merge[0, :, :]
        test = test.fillna(0)
        # indexing: start:stop:step
        test[1:test.shape[0], :] = test[0:test.shape[0] - 1, :].values + test[1:test.shape[0], :].values  # down
        test[1:test.shape[0]:2, 1:test.shape[1]:2] = test[0:test.shape[0] - 1:2, 0:test.shape[1] - 1:2].values + test[1:
                                                                                                                      test.shape[
                                                                                                                          0]:2,
                                                                                                                 1:
                                                                                                                 test.shape[
                                                                                                                     1]:2].values  # down + right
        test[0:test.shape[0]:2, 1:test.shape[1]:2] = test[0:test.shape[0]:2, 0:test.shape[1] - 1:2].values + test[
                                                                                                             0:
                                                                                                             test.shape[
                                                                                                                 0]:2,
                                                                                                             1:
                                                                                                             test.shape[
                                                                                                                 1]:2].values  # right

        test = numpy.array(test)  # for checking

        # at the moment tp_merge only the upper left cell is filled with values

        #     2km
        ###############
        # Value## NaN ## 1km
        #     ##     ##
        ###############
        # NaN ## NaN ## 1km
        #     ##     ##
        ###############
        # 1km    1km
        # scaling up the test from above
        tp_merge = tp_merge.fillna(0)
        tp_merge[:, 1:tp_merge.shape[1], :] = tp_merge[:, 0:tp_merge.shape[1] - 1, :].values + tp_merge[:,
                                                                                               1:tp_merge.shape[1],
                                                                                               :].values  # down
        tp_merge[:, 1:tp_merge.shape[1]:2, 1:tp_merge.shape[2]:2] = tp_merge[:, 0:tp_merge.shape[1] - 1:2,
                                                                    0:tp_merge.shape[2] - 1:2].values + tp_merge[:,
                                                                                                        1:
                                                                                                        tp_merge.shape[
                                                                                                            1]:2,
                                                                                                        1:
                                                                                                        tp_merge.shape[
                                                                                                            2]:2].values  # down + right
        tp_merge[:, 0:tp_merge.shape[1]:2, 1:tp_merge.shape[2]:2] = tp_merge[:, 0:tp_merge.shape[1]:2,
                                                                    0:tp_merge.shape[2] - 1:2].values + tp_merge[:,
                                                                                                        0:
                                                                                                        tp_merge.shape[
                                                                                                            1]:2,
                                                                                                        1:
                                                                                                        tp_merge.shape[
                                                                                                            2]:2].values  # right

        # testing the result
        test = tp_merge[0, :, :]
        test = numpy.array(test)  # for checking

        tp_merge_div = tp_merge

        # testing the result
        test = tp_merge_div[0, :, :]
        test = numpy.array(test)  # for checking

        # to do: add all the attributes from the original dataset
        # some of the attributes may have to be changed to be correct
        tp.attrs

        tp_merge_div.attrs = tp.attrs
        print('made it till here')
        print("Current Working Directory ", os.getcwd())
        c_wd = os.getcwd()
        os.chdir(os.path.join(output_folder, str(y) + '_netcdf'))
        tp_merge_div.to_netcdf(f.replace('.grb', '.nc'), 'w', 'NETCDF4', engine='netcdf4')




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


# reproject data
import netCDF4
import numpy as np
import cdo

indir = ('/Volumes/JAN/COSMO/2007_netcdf/TOTAL_PRECIPITATION.SFC.200701.nc')
f = netCDF4.Dataset(indir)
print(f)
#lat = f.variables['rlat']
#lon = f.variables['rlon']
lat = f.variables['x']

from pyproj import CRS
crs = CRS.from_proj4("+proj=ob_tran +o_proj=longlat +o_lon_p=-162 +o_lat_p=39.25 +lon_0=180 +to_meter=0.01745329")
crs_4326 = CRS.from_epsg(4326) # WKID WGS84
crs_4326
from pyproj import Transformer
transformer = Transformer.from_crs(crs_4326, crs)
transformer.transform(-23.40299988, -28.403) #lat, lon vom südwestlichsten teil

#until here we are only able to reproject one grid point -> with this method we could try to create lon/lat tubles and put them in the transform step above
#the following part is from https://rasterio.readthedocs.io/en/stable/topics/reproject.html and could be the solution the reproject entire rasters


# reproject with rasterio
import rasterio
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling

with rasterio.Env():

    # As source: a 512 x 512 raster centered on 0 degrees E and 0
    # degrees N, each pixel covering 15".
    rows, cols = src_shape = (1447, 1559)
    d = 1.0/240 # decimal degrees per pixel
    # The following is equivalent to
    # A(d, 0, -cols*d/2, 0, -d, rows*d/2).
    src_transform = A.translation(-cols*d/2, rows*d/2) * A.scale(d, -d)
    src_crs = {'init': 'EPSG:4328'}
    src_crs = crs
    source = numpy.ones(src_shape, numpy.uint8)*255

    # Destination: a 1024 x 1024 dataset in Web Mercator (EPSG:3857)
    # with origin at 0.0, 0.0.
    dst_shape = (1447, 1559)
    dst_transform = [-237481.5, 425.0, 0.0, 237536.4, 0.0, -425.0]
    dst_transform = transformer
    dst_crs = {'init': 'EPSG:4326'}
    destination = numpy.zeros(dst_shape, numpy.uint8)

    reproject(
        source,
        destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest)

    # Assert that the destination is only partly filled.
    assert destination.any()
    assert not destination.all()