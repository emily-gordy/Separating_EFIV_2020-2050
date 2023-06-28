#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:14:36 2022

@author: emgordy
"""

# preprocess the SST data into annualmeans

import xarray as xr
import numpy as np
import glob

path = "/Users/emgordy/Documents/CESM2-LE/regridded_all/"
path2 = "/Users/emgordy/Documents/Experiments/CESM2-LE-prediction/regridded/"

modelstartyears = np.arange(1001,1201,10)

baselineanom1 = 1981 # year range over which to calculate SST anom
baselineanom2 = 2010

LENSsst = []    
for istart, start in enumerate(modelstartyears):

    LENSstr = glob.glob(path+"CESM2*SST*-"+str(start)+".*.nc")[0]
    print(LENSstr)
    LENSds = xr.open_dataset(LENSstr)
    SSTloop = LENSds.SST
    SSTloop.assign_coords({"variant": istart+1})
    LENSsst.append(SSTloop)

modelstartyears1 = np.array([1231,1251,1281,1301])
ensnums = np.arange(1,21)
count=istart+1

for istart, start in enumerate(modelstartyears1):
    for ens in ensnums:
        ensnum = "{:03d}".format(ens)
        LENSstr = glob.glob(path+"CESM2*SST*-"+str(start)+"."+ensnum+".*.nc")[0]
        print(LENSstr)
        LENSds = xr.open_dataset(LENSstr)
        SSTloop = LENSds.SST
        SSTloop.assign_coords({"variant": count})
        LENSsst.append(SSTloop)  
        count +=1
        print(count)

LENSsst = xr.concat(LENSsst,dim="variant")

lat = LENSsst.lat
lon = LENSsst.lon

# mask out extra land to nans
lmstr = glob.glob(path2+"sftlf*72x36*nc")[0]
lmds = xr.open_dataset(lmstr)
lm = lmds.sftlf

LENSsst = xr.where(lm<20,LENSsst,np.nan)

#%% annual means

annualmean = LENSsst.groupby("time.year").mean()
annualmean = annualmean.assign_coords({"variant" : np.arange(1,101)})

baselinemean = annualmean.sel(year=slice(baselineanom1,baselineanom2)).mean(dim="year")
annualmeananom = annualmean-baselinemean
#%%
annualmeananom = annualmeananom.rename("SST")
annualmeananom.to_netcdf("data/LENS_SSTannualmean.nc")

annualmean = annualmean.rename("SST")
annualmean.to_netcdf("data/LENS_SSTannualmean_fullfield.nc")