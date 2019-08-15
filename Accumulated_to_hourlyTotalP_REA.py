


import xarray as xr
import netCDF4
import pandas as pd
import numpy as np
import tkinter


from tkinter import filedialog
from tkinter import *

root = Tk()
file_netcdf =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("netcdf4 files","*.nc"),("all files","*.*")))
print (file_netcdf)


ds = xr.open_dataset(file_netcdf)

time = ds.__getitem__('time')
tp = ds.__getitem__('tp')


date = pd.to_datetime(time.values)
# date.hour
tp_2 = tp.copy(deep = True)
zeitpunkt_h2 = tp[1,:,:].values

# alle 6 Werte ein Raster mit Niederschlagswerten -> 1, 7, 13, 19

# Niederschlagsanpassung gemäss Readme --> Prinzip bei allen Rastern auser die der Stunden 1,7,13,19 wird das
# vorangehnde Raster subtrahiert
# etwas strange ist das mit der Stepdimension -> durch pröbeln die richtige Dimension ermittelt.
for counter, h in enumerate(date.hour):
    print('counter: ' + str(counter) + ', hour: ' + str(h))
    if h in [2, 8, 14, 20]: # Wenn Stunde 2 Uhr, dann muss 2-1 gerechnet werden
        #tp_2[counter,:,:].values = tp[counter,:,:].values - tp[counter-1,:,:].values
        tp_2[counter,:,:] = tp[counter,:,:] - tp[counter-1,:,:]
        print(str(counter) + ' geändert')
    if h in [3, 9, 15, 21]: # 3 Uhr --> 3-2
        tp_2[counter,:,:] = tp[counter,:,:] - tp[counter-1,:,:]
        print(str(counter) + ' geändert')
    if h in [4, 10, 16, 22]: # 3 Uhr --> 3-2
        tp_2[counter,:,:] = tp[counter,:,:] - tp[counter-1,:,:]
        print(str(counter) + ' geändert')
    if h in [5, 11, 17, 23]: # 3 Uhr --> 3-2
        tp_2[counter,:,:] = tp[counter,:,:] - tp[counter-1,:,:]
        print(str(counter) + ' geändert')
    if h in [6, 12, 18, 0]: # 3 Uhr --> 3-2
        tp_2[counter,:,:] = tp[counter,:,:] - tp[counter-1,:,:]
        print(str(counter) + ' geändert')

file_export =  filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("netcdf4 files","*.nc"),("all files","*.*")))
tp_2.to_netcdf(file_export)




