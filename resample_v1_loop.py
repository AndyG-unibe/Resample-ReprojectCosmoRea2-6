# Script to resample, convert and reproject COSMO precipitation rasters

# Authors
# Andy Gschwind, Jan Liechti

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
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Reproject Function from https://www.earthdatascience.org/courses/earth-analytics-python/lidar-raster-data/reproject-raster/
def reproject_et(inpath, outpath, new_crs):
    dst_crs = new_crs # CRS for web meractor

    with rio.open(inpath) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(outpath, 'w+', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


years = numpy.arange(2007, 2014, 1).tolist()
url = 'ftp://ftp.meteo.uni-bonn.de/pub/reana/COSMO-REA2/TOT_PREC/'

output_folder = '/Volumes/Untitled/COSMO/REA2'

"""
file = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project/Data/TOTAL_PRECIPITATION.SFC.200701.grb'
file_netcdf = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project/Data/test.nc'
path_export_folder = r'/Users/janliechti/Google Drive/UNI/FS19/Geographie/Gedatenanalyse_u_Modellierung/Project_COSMO/PyCharm_project'
ds = xr.open_dataset(file, engine='cfgrib')
"""
# COSMO Rea6
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
        # every 6 values a raster with precipitation values -> positions: 1, 7, 13, 19
        # calculation of the hourly precipitation values according to metadata of COSMO REA6
        # figured step dimension out through trial & error
        # print statments are in german and just for control of loop
        # 'geändert' means changed
        # 'kopiert' means copied

        for counter, h in enumerate(date.hour):
            print('counter: ' + str(counter) + ', hour: ' + str(h))
            if h in [2, 8, 14, 20]:  # When hour ==2 the first raster has to be subtracted
                tp_2[counter, 1, :, :] = tp[counter, 1, :, :] - tp[counter - 1, 0, :, :]
                print(str(counter) + ' geändert')
            if h in [3, 9, 15, 21]:  # 3 Uhr --> 3-2
                tp_2[counter, 2, :, :] = tp[counter, 2, :, :] - tp[counter - 1, 1, :, :]
                print(str(counter) + ' geändert')
            if h in [4, 10, 16, 22]:
                tp_2[counter, 3, :, :] = tp[counter, 3, :, :] - tp[counter - 1, 2, :, :]
                print(str(counter) + ' geändert')
            if h in [5, 11, 17, 23]:
                tp_2[counter, 4, :, :] = tp[counter, 4, :, :] - tp[counter - 1, 3, :, :]
                print(str(counter) + ' geändert')
            if h in [6, 12, 18, 0]:
                tp_2[counter, 5, :, :] = tp[counter, 5, :, :] - tp[counter - 1, 4, :, :]
                print(str(counter) + ' geändert')

        # create Data array in desired size without step-Dimension
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
            if h in [2, 8, 14, 20]:
                tmp[counter, :, :] = tp[counter, 1, :, :] - tp[counter - 1, 0, :, :]
                print(str(counter) + ' geändert')
            if h in [3, 9, 15, 21]:
                tmp[counter, :, :] = tp[counter, 2, :, :] - tp[counter - 1, 1, :, :]
                print(str(counter) + ' geändert')
            if h in [4, 10, 16, 22]:
                tmp[counter, :, :] = tp[counter, 3, :, :] - tp[counter - 1, 2, :, :]
                print(str(counter) + ' geändert')
            if h in [5, 11, 17, 23]:
                tmp[counter, :, :] = tp[counter, 4, :, :] - tp[counter - 1, 3, :, :]
                print(str(counter) + ' geändert')
            if h in [6, 12, 18, 0]:
                tmp[counter, :, :] = tp[counter, 5, :, :] - tp[counter - 1, 4, :, :]
                print(str(counter) + ' geändert')

        # make a double sized array and then merge (overlay)
        data = numpy.ndarray([tmp.shape[0], 2 * tmp.shape[1] - 1, 2 * tmp.shape[2] - 1])
        data[:] = None
        coords_y = numpy.arange(min(ds.__getitem__('y').values), max(ds.__getitem__('y').values) + 0.5,
                                0.5)  # make coords in 0.5 instead of 1 step --> double size
        coords_x = numpy.arange(min(ds.__getitem__('x').values), max(ds.__getitem__('x').values) + 0.5, 0.5)
        tmp_x2 = xr.DataArray(data, dims={'time': 743, 'y': 2 * 780, 'x': 2 * 724},
                              coords=[ds.__getitem__('time').values, coords_y, coords_x])
        tmp_x2 = tmp_x2.to_dataset(name = 'tp')
        tmp = tmp.to_dataset(name= 'tp')
        tmp_merge = xr.merge([tmp_x2, tmp], compat='no_conflicts')
        tp_merge = tmp_merge.__getitem__('tp')

        del ( tmp_x2, tmp_merge, data, ds)
        # for better understanding here a test example
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


        tp_merge_div.attrs = tp.attrs
        print("Current Working Directory ", os.getcwd())
        c_wd = os.getcwd()
        os.chdir(os.path.join(output_folder, str(y) + '_netcdf'))
        tp_merge_div.to_netcdf(f.replace('.grb', '.nc'), 'w', 'NETCDF4', engine='netcdf4') # save raster as Netcdf
        # Reprojection of the data
        f_tmp = f.replace('.grb', '_reprojected.nc')
        reproject_et(os.path.join(output_folder, str(y) + '_netcdf', f.replace('.grb', '.nc')),
                     os.path.join(output_folder, str(y) + '_netcdf', f_tmp), 'EPSG:3857') # Web Mercator reproject

# COSMO Rea2
years = numpy.arange(1995, 2015, 1).tolist()
url = 'ftp://ftp.meteo.uni-bonn.de/pub/reana/COSMO-REA6/HOURLY/TOTAL.PRECIPITATION/'
output_folder = '/Volumes/Untitled/COSMO/REA6'

# for testing the loop
y = years[0]
f = 'TOTAL.PRECIPITATION.200801.grb'

for y in years:
    # download files directly via python
    ftp = ftplib.FTP(r'ftp.meteo.uni-bonn.de')
    ftp.login()
    ftp.cwd('/pub/reana/COSMO-REA6/HOURLY/TOTAL.PRECIPITATION' )
    grb_list = ftp.nlst()  # List files in folder
    grb_list_grb = []
    for i in grb_list: # select all .grb files, leave the .idx out
        if '.idx' not in i:
            grb_list_grb.append(i)
    grb_list = grb_list_grb
    os.makedirs(os.path.join(output_folder, str(y) + '_grib'))
    os.makedirs(os.path.join(output_folder, str(y) + '_netcdf'))
    for f in grb_list:
        wget.download(url + '/' + f, os.path.join(output_folder, str(y)  + '_grib', f))
        ds = xr.open_dataset(os.path.join(output_folder, str(y) + '_grib', f), engine='cfgrib')

        time = ds.__getitem__('time')  # Timesteps
        tp = ds.__getitem__('tp')
        steps = ds.__getitem__('step')



        date = pd.to_datetime(time.values)
        # date.hour
        zeitpunkt_h2 = tp[1, 1, :].values
        # No processing of cumulative precipitation values necessary --> already hourly

        # create Data array
        # data = numpy.full((743, 780, 724), None)
        data = numpy.ndarray([tp.shape[0], tp.shape[1], tp.shape[2]])
        data[:] = None
        tmp = xr.DataArray(data, dims={'time': tp.shape[0], 'y': tp.shape[1], 'x': tp.shape[2]},
                           coords=[ds.__getitem__('time').values, ds.__getitem__('y').values,
                                   ds.__getitem__('x').values])


        # make a tripple sized array and then merge (overlay)
        data = numpy.ndarray([tp.shape[0], 3 * tp.shape[1] - 1, 3 * tp.shape[2] - 1])
        data[:] = None
        # ev. hier umprjizieren
        coords_y = numpy.arange(min(ds.__getitem__('y').values), max(ds.__getitem__('y').values) + 1/3,
                                1/3)  # make coords in 1/3 instead of 1 step --> tripple size
        coords_y[0:len(coords_y):3] = numpy.arange(0, max(ds.__getitem__('y').values)+1, 1)
        coords_x = numpy.arange(min(ds.__getitem__('x').values), max(ds.__getitem__('x').values) + 1/3, 1/3)
        coords_x[0:len(coords_x):3] = numpy.arange(0, max(ds.__getitem__('x').values) + 1, 1)
        tmp_x2 = xr.DataArray(data, dims={'time': 744, 'y': 3 * 824, 'x': 3 * 848},
                              coords=[ds.__getitem__('time').values, coords_y, coords_x])
        tmp_x2 = tmp_x2.to_dataset(name='tp')
        tmp = tmp.to_dataset('tp')
        tmp_merge = xr.merge([tmp_x2, tmp], compat='no_conflicts')
        tp_merge = tmp_merge.__getitem__('tp')
        # from here on i couldn't test the script, my computer just hasn't enough recources to compute the merge
        # all following commands are just estimated

        del (coords_x, coords_y, tmp_x2, tmp_merge, data, ds)
        # Working 2D Example of filling the gaps --> important: fill nan with 0
        test = tp_merge[0, :, :]

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
                                                                                               :].values  # down 1
        tp_merge[:, 2:tp_merge.shape[1], :] = tp_merge[:, 0:tp_merge.shape[1] - 1, :].values + tp_merge[:,
                                                                                               2:tp_merge.shape[1],
                                                                                               :].values  # down 2
        tp_merge[:, 1:tp_merge.shape[1]:2, 1:tp_merge.shape[2]:2] = tp_merge[:, 0:tp_merge.shape[1] - 1:2,
                                                                    0:tp_merge.shape[2] - 1:2].values + tp_merge[:,
                                                                                                        1:
                                                                                                        tp_merge.shape[
                                                                                                            1]:2,
                                                                                                        1:
                                                                                                        tp_merge.shape[
                                                                                                            2]:2].values  # down + right
        tp_merge[:, 1:tp_merge.shape[1]:3, 1:tp_merge.shape[2]:3] = tp_merge[:, 0:tp_merge.shape[1] - 1:3,
                                                                    0:tp_merge.shape[2] - 1:3].values + tp_merge[:,
                                                                                                        1:
                                                                                                        tp_merge.shape[
                                                                                                            1]:3,
                                                                                                        1:
                                                                                                        tp_merge.shape[
                                                                                                            2]:3].values  # down + right 2
        tp_merge[:, 0:tp_merge.shape[1]:2, 1:tp_merge.shape[2]:2] = tp_merge[:, 0:tp_merge.shape[1]:2,
                                                                    0:tp_merge.shape[2] - 1:2].values + tp_merge[:,
                                                                                                        0:
                                                                                                        tp_merge.shape[
                                                                                                            1]:2,
                                                                                                        1:
                                                                                                        tp_merge.shape[
                                                                                                            2]:2].values  # right
        tp_merge[:, 0:tp_merge.shape[1]:3, 1:tp_merge.shape[2]:3] = tp_merge[:, 0:tp_merge.shape[1]:3,
                                                                    0:tp_merge.shape[2] - 1:3].values + tp_merge[:,
                                                                                                        0:
                                                                                                        tp_merge.shape[
                                                                                                            1]:3,
                                                                                                        1:
                                                                                                        tp_merge.shape[
                                                                                                            2]:3].values  # right 2




        tp_merge_div.attrs = tp.attrs
        print('made it till here')
        print("Current Working Directory ", os.getcwd())
        c_wd = os.getcwd()
        os.chdir(os.path.join(output_folder, str(y) + '_netcdf'))
        tp_merge_div.to_netcdf(f.replace('.grb', '.nc'), 'w', 'NETCDF4', engine='netcdf4')
        # Reprojection of the data
        f_tmp = f.replace('.grb', '_reprojected.nc')
        reproject_et(os.path.join(output_folder, str(y) + '_netcdf', f.replace('.grb', '.nc')),
                     os.path.join(output_folder, str(y) + '_netcdf', f_tmp), 'EPSG:3857')  # Web Mercator reproject


# Try and error content
"""
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
# import cdo

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
transformer.transform(-23.40299988, 0) #lat, lon vom südwestlichsten teil
coords_transformed = []
for index, i in enumerate(coords_x):
    coords_transformed.append(transformer.transform(i, 0)[0])




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
    assert not destination.all()"""