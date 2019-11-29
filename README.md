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
### Description in detail

In this section all the different processing steps are explained.

Download: 
At first the .grb files are downloaded directly from the FTP-server. 
* The COSMO REA2 we get from here: ftp://ftp.meteo.uni-bonn.de/pub/reana/COSMO-REA2/TOT_PREC/
* The COSMO REA6 we get from here: ftp://ftp.meteo.uni-bonn.de/pub/reana/COSMO-REA6/HOURLY/TOTAL.PRECIPITATION/
Then the files are stored in a newly created directory. 

Processing: 
The processing of the .grb rasters happens mostly within a loop. The procedure happes as follows: 
* Opening the files with xarray
* Extracting the precipitation values
* Transforming cumulative precipitation to hourly precipitation (only for COSMO REA2) values (according to metadata: ftp://ftp.meteo.uni-bonn.de/pub/reana/COSMO-REA2/README_COSMO_REA2.txt)
* For the COSMO REA2: Create a double sized empty array for the resampling -> cell size of 1km instead of 2km
* Overlay the double sized array with the original array. Then fill up the missing precipitation values. Because the unit of the precipitation is mm/per cell area no adjustment of the precipitation values is necessary. 
* Export of the new sized array as Netcdf file. 
***
### Difficulties

* We can not a plausibilisation of the output data with a reference. Thins means that we can't determine obvious mistakes that may occur during the processing. 
* We had a lot of difficulties with the operating Systems. With Windows the .grb file processing within python wasn't possible, so we decided to use Mac OSX for doing the processing. This made things complicated because only one of us had a Mac computer. Using a unix based OS should to the job as well. 
* Reprojection: We are not shure if we did the right thing. At the end we ended up recalculating the axis of the raster. 

### Acknowledgements

We would like to give thanks to our supporters during the semesters course Pascal Horton, Andreas Zischg and Jorge Ramirez. For the instructions and friendly help for our little python project we would like to thank Martina Kauzlaric.
