# 2019-precipitation-discretization

Resampling and Reprojecting of Cosmo-Rea2 & Cosmo-Rea6 precipitation reanalysis data

Authors:

...Andy Gschwind, 14-111-900

...Jan Liechti, 14-

This project is our homework submission for the seminar Geodata analysis and modeling at the University of Bern in Switzerland. 

### Overview

This script should improve the work with Cosmo-Rea precipitation datasets. It contains the following processing steps:

* Downloading datasets directly via Python

* Transforming the cummulative precipitation values to hourly precipitation values

* Resampling of Cosmo-Rea2 (2x2km grid) to a 1x1km grid and Cosmo-Rea6 (6x6km grid) to a 2x2km grid

* Reprojecting the grids from a rotated pole grid to WGS84 projection

* Saving the resulting files as Netcdf-file


### Installation

You have to install to following codes like that:

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


### Description 

In this section all the different processing steps are explained.

