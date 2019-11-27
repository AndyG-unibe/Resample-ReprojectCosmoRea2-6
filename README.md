# 2019-precipitation-discretization

Resampling and Reprojecting of Cosmo-Rea2 & Cosmo-Rea6 precipitation reanalysis data

##### Authors

* Andy Gschwind, 14-111-900

* Jan Liechti, 12-924-346

This project is our homework submission for the seminar Geodata analysis and modeling at the University of Bern in Switzerland. 

***
### Overview

This script improves the work with Cosmo-Rea precipitation datasets. It contains the following processing steps:

* Downloading datasets directly from ftp-server via Python

* Transforming the cummulative precipitation values to hourly precipitation values

* Resampling of Cosmo-Rea2 (2x2km grid) to a 1x1km grid and Cosmo-Rea6 (6x6km grid) to a 2x2km grid

* Reprojecting the grids from a rotated pole grid to WGS84 projection

* Saving the resulting files as Netcdf-file

***
### Installation

The script was develepped to run on an OSX-System. To ensure compatibility one has install XCode and homebrew: https://brew.sh/index_de
Then "$ brew install eccodes" has to be executed in the terminal to install eccodes. Furthermore when the Cfgrib-package can not be installed by pip one has to download the package manually and move it to the IDE side package directory. Cfgrib-package can be downloaded here: https://github.com/ecmwf/cfgrib

Also you have to make shure to install the following packages:

xarray, cf2cdm, cfgrib, os, netCDF4, numpy, pandas, rasterio, wget, ftplib


***
### Description 

In this section all the different processing steps are explained.

