#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:29:43 2022

@author: emgordy
"""

# this will eventually be a pre-processing function but for now it is a script

import xarray as xr
import numpy as np
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from scipy import sparse
from scipy.sparse.linalg import eigs

import os
path=os.path.realpath(os.path.dirname(__file__))
path += "/../data/"

# specify a lead year, averaging length and date range and output the SST maps
# for that

# note lead years + averaging in the lookback sense e.g. lead year 5 with 5 year averaging
# is the average of years -10 to -6 inclusive

# daterange is for PREDICTIONS so if daterange[0] is 1980 with ly 5 and 5 year averaging
# then should return SST anoms with first entry average SST in 1971-1975 inclusive
# This is so there shouldn't be an overlap between lead information and the prediction

# This means that ly = 1 means we predict next year

def SSTinput_nft(ly, averaginglength, daterange, trendlength):
    # path = "data/"
    ds = xr.open_dataset(path+"LENS_SSTannualmean.nc")

    # timeseries ends in 2101 so can't predict too far ahead

    # if (daterange[1]+trendlength)>2100:
    daterange_use = [daterange[0], daterange[1]-trendlength]

    SST = ds.SST
    SSTmean = SST.rolling(year=averaginglength).mean()
    SSTcut = SSTmean.sel(year=slice(daterange_use[0]-ly, daterange_use[1]-ly))

    SSTcutmean = SSTcut.mean(dim="variant")
    SSTcut = SSTcut-SSTcutmean
    SSTcut = SSTcut.transpose("year", "variant", "lat", "lon")
    return SSTcut


def SSTinput_ft(ly, averaginglength, daterange, trendlength):
    # path = "data/"
    ds = xr.open_dataset(path+"LENS_SSTannualmean.nc")

    # timeseries ends in 2101 so can't predict too far ahead

    # if (daterange[1]+trendlength)>2100:
    daterange_use = [daterange[0], daterange[1]-trendlength]

    SST = ds.SST
    SSTmean = SST.rolling(year=averaginglength).mean()
    SSTcut = SSTmean.sel(year=slice(daterange_use[0]-ly, daterange_use[1]-ly))

    SSTcutmean = SSTcut.mean(dim="variant")

    onesmat = np.ones(SSTcut.shape)
    onesda = xr.DataArray(
        data=onesmat,
        dims=["lat", "lon", "variant", "year"],
        coords=[SSTcut.lat, SSTcut.lon, SSTcut.variant, SSTcut.year]
    )
    # so output has dimensions lat x lon x variant x year
    SSTcutmean = SSTcutmean*onesda
    SSTcutmean = SSTcutmean.transpose("year", "variant", "lat", "lon")
    return SSTcutmean



# %% now make predictions vector


def SSToutput(trendlength=10, lat1=-50, lat2=-30, lon1=170, lon2=180, daterange=[1980, 2100]):
    # load in SST annual means
    # path = "data/"
    ds = xr.open_dataset(path+"LENS_SSTannualmean.nc")

    # grab region to calculate over
    SST = ds.SST
    SSTregion = SST.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2))
    year = np.asarray(ds.year)

    weights = np.cos(np.deg2rad(SSTregion.lat))
    SSTregionmean = np.asarray(SSTregion.weighted(
        weights).mean(dim=("lat", "lon")))

    # calculate decadal trends in each ensemble member
    tvec = np.arange(0, 1, 1/trendlength)
    timeshort = year[:-trendlength]
    variant = np.arange(1, 101)

    ntime = len(year)

    SSTtrendvec = []
    yintvec = []

    for itime in range(ntime-int(trendlength)):
        # so dimensions are right in the polyfit
        loopvec = np.transpose(SSTregionmean[:, itime:itime+int(trendlength)])
        [SSTtrend, yint] = np.polyfit(tvec, loopvec, deg=1)
        SSTtrendvec.append(SSTtrend)
        yintvec.append(yint)

    SSTtrendvec = np.asarray(SSTtrendvec)
    yintvec = np.asarray(yintvec)

    da = xr.DataArray(
        data=SSTtrendvec,
        dims=["year", "variant"],
        coords=dict(
            year=timeshort,
            variant=variant
        ),
        attrs=dict(
            description="SST trend",
            units="degC/decade",
        ),
    )

    da = da.rename("SST_trend")
    # cut data and remove burnt edges

    # if (daterange[1]+trendlength)>2100:
    daterange_use = [daterange[0], daterange[1]-trendlength]
    SSTout = da.sel(year=slice(daterange_use[0], daterange_use[1]))

    return SSTout

def SSToutput_all(trendlength=10, lat1=-50, lat2=-30, lon1=170, lon2=180, daterange=[1980, 2100]):
    # load in SST annual means
    # path = "data/"
    ds = xr.open_dataset(path+"LENS_SSTannualmean_fullfield.nc")

    # grab region to calculate over
    SST = ds.SST
    SSTregion = SST.sel(year=slice(daterange[0],daterange[1]),lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    year = SSTregion.year

    weights = np.cos(np.deg2rad(SSTregion.lat))
    SSTregionmean = np.asarray(SSTregion.weighted(
        weights).mean(dim=("lat", "lon")))

    
    # calculate decadal trends in each ensemble member
    tvec = np.arange(0,1,1/trendlength)
    timeshort = year[:-trendlength]
    variant = np.arange(1, 101)

    ntime = SSTregionmean.shape[1]

    SSTtrendvec = []
    yintvec = []

    for itime in range(ntime-int(trendlength)):
        # so dimensions are right in the polyfit
        loopvec = np.transpose(SSTregionmean[:, itime:itime+int(trendlength)])
        [SSTtrend, yint] = np.polyfit(tvec, loopvec, deg=1)
        SSTtrendvec.append(SSTtrend)
        yintvec.append(yint)

    yintvec = np.asarray(yintvec)

    da = xr.DataArray(
        data=yintvec,
        dims=["year", "variant"],
        coords=dict(
            year=timeshort,
            variant=variant
        ),
        attrs=dict(
            description="SST yint",
            units="degC",
        ),
    )

    da = da.rename("SST_yint")    
    yint_out = da-273.15 # celsius reigns supreme
    SSTregionmean = SSTregionmean-273.15

    return SSTregionmean,year,yint_out

def SSToutput_persistence(trendlength=10, lat1=-50, lat2=-30, lon1=170, lon2=180, daterange=[1980, 2100]):
    # load in SST annual means
    # path = "data/"
    ds = xr.open_dataset(path+"LENS_SSTannualmean.nc")

    # grab region to calculate over
    SST = ds.SST
    SSTregion = SST.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2))
    year = np.asarray(ds.year)

    weights = np.cos(np.deg2rad(SSTregion.lat))
    SSTregionmean = np.asarray(SSTregion.weighted(
        weights).mean(dim=("lat", "lon")))

    # calculate decadal trends in each ensemble member
    tvec = np.arange(0, 1, 1/trendlength)
    timeshort = year[:-trendlength]
    variant = np.arange(1, 101)

    ntime = len(year)

    SSTtrendvec = []
    yintvec = []

    for itime in range(ntime-int(trendlength)):
        # so dimensions are right in the polyfit
        loopvec = np.transpose(SSTregionmean[:, itime:itime+int(trendlength)])
        [SSTtrend, yint] = np.polyfit(tvec, loopvec, deg=1)
        SSTtrendvec.append(SSTtrend)
        yintvec.append(yint)

    SSTtrendvec = np.asarray(SSTtrendvec)
    yintvec = np.asarray(yintvec)

    da = xr.DataArray(
        data=SSTtrendvec,
        dims=["year", "variant"],
        coords=dict(
            year=timeshort,
            variant=variant
        ),
        attrs=dict(
            description="SST trend",
            units="degC/decade",
        ),
    )

    da = da.rename("SST_trend")
    # cut data and remove burnt edges

    # if (daterange[1]+trendlength)>2100:
    daterange_use = [daterange[0]-trendlength, daterange[1]-2*trendlength]
    SSTout = da.sel(year=slice(daterange_use[0], daterange_use[1]))

    return SSTout

# %% now function for train/val/test and also flatten the lat/lon


def memsplit(matin, memchoose):

    matin = np.asarray(matin)
    matdims = matin.shape

    nmems = len(memchoose)

    if len(matdims) > 3:
        matout = np.empty((matdims[0]*nmems, matdims[2]*matdims[3]))
    elif len(matdims)==3:
        matout = np.empty((matdims[0]*nmems,matdims[2]))
    else:
        matout = np.empty(matdims[0]*nmems)

    for imem, mem in enumerate(memchoose):
        if len(matdims) > 3:
            Xmem = matin[:, mem, :, :]
            matout[matdims[0]*imem:matdims[0] *
                   (imem+1), :] = np.reshape(Xmem, (matdims[0], matdims[2]*matdims[3]))
        # elif len(matdims)==3:
        #     matout[matdims[0]*imem:matdims[0]*(imem+1)]
        else:
            matout[matdims[0]*imem:matdims[0]*(imem+1)] = matin[:, mem]

    matout[np.isnan(matout)] = 0

    return matout

# %% and a function for making the Y data into categorical terciles


def y_onehot(Yin, ltercile, utercile):
    Yout = np.empty(len(Yin))

    Yout[Yin < ltercile] = 0
    Yout[(Yin >= ltercile) & (Yin < utercile)] = 1
    Yout[Yin >= utercile] = 2

    Yout = to_categorical(Yout, num_classes=3)

    return Yout


# %% caluclate ucutoff from training data but only up for given ranbe

def calc_ucutoff(SSTout, trainens, date1, date2):

    SSTfunc = SSTout.sel(year=slice(date1, date2)).isel(variant=trainens)
    ucutoff = SSTfunc.quantile(q=0.66666666667, dim=("year", "variant"))

    # these have to be arrays and I can't remember why
    return np.asarray(ucutoff)


def calc_lcutoff(SSTout, trainens, date1, date2):

    SSTfunc = SSTout.sel(year=slice(date1, date2)).isel(variant=trainens)
    lcutoff = SSTfunc.quantile(q=0.33333333333, dim=("year", "variant"))

    return np.asarray(lcutoff)


def calc_ucutoff_q(SSTout, trainens, date1, date2, q):

    SSTfunc = SSTout.sel(year=slice(date1, date2)).isel(variant=trainens)
    ucutoff = SSTfunc.quantile(q=q, dim=("year", "variant"))

    # these have to be arrays and I can't remember why
    return np.asarray(ucutoff)


def calc_lcutoff_q(SSTout, trainens, date1, date2, q):

    SSTfunc = SSTout.sel(year=slice(date1, date2)).isel(variant=trainens)
    lcutoff = SSTfunc.quantile(q=q, dim=("year", "variant"))

    return np.asarray(lcutoff)


def enslists(ensnums):
    nenssets = 10
    ensout = []
    for i in range(nenssets):
        if len(ensnums) > 1:
            for iens in range(len(ensnums)):
                ensloop = [2*ensnums[iens]+nenssets *
                           i, 2*ensnums[iens]+1+nenssets*i]
                ensout.append(ensloop)
        else:
            ensloop = [2*ensnums[0]+nenssets*i, 2*ensnums[0]+1+nenssets*i]
            ensout.append(ensloop)

    ensout = np.asarray(ensout).flatten()
    return ensout

    

def Nino34(daterange=[1980, 2100]):
    # load in SST annual means
    # path = "data/"
    ds = xr.open_dataset(path+"LENS_SSTannualmean_fullfield.nc")
    
    lat1 = -5
    lat2 = 5
    lon1 = 170
    lon2 = 240
    
    # grab region to calculate over
    SST = ds.SST
    SSTregion = SST.sel(year=slice(daterange[0],daterange[1]),lat=slice(lat1,lat2),lon=slice(lon1,lon2))

    fr = SSTregion.mean(dim="variant")
    SST_IV = SSTregion-fr
    
    
    weights = np.cos(np.deg2rad(SST_IV.lat))
    nino34 = SST_IV.weighted(
        weights).mean(dim=("lat", "lon"))

    return nino34  
    
def PDO(daterange=[1980,2100]):
    
    ds = xr.open_dataset(path+"LENS_SSTannualmean_fullfield.nc")

    lon = ds.lon
    lat = ds.lat
    year = ds.year
    variant = ds.variant
    sst = ds.SST
    
    sstfr = sst.mean(dim="variant")
    sstnofr = sst-sstfr
    
    Paclon1 = 110
    Paclon2 = 260
    Paclat1 = 20
    Paclat2 = 60
    
    Paclat = lat.sel(lat=slice(Paclat1,Paclat2))
    Paclon = lon.sel(lon=slice(Paclon1,Paclon2))
    
    Paclonxlat = np.meshgrid(Paclon,Paclat)[1]
    Pacweights = np.sqrt(np.cos(Paclonxlat*np.pi/180))
    
    PacificSST = np.asarray(sstnofr.sel(lon=slice(Paclon1,Paclon2),lat=slice(Paclat1,Paclat2)))
    
    PacificSST_flat = np.reshape(PacificSST,(8,30,100*252))
    
    PacificSSTw = PacificSST_flat*Pacweights[:,:,np.newaxis]
    
    Pacnoland = PacificSSTw[~np.isnan(PacificSST_flat[:,:,0]),:]
    PacCov = np.cov(Pacnoland)
    
    PacCov_s = sparse.csc_matrix(PacCov)
    
    eigval,evec = eigs(PacCov_s,1)
    evec = np.squeeze(np.real(evec))
    
    if evec[0]>0:
        evec = -1*evec
    
    PacEOF = np.empty((Paclat.shape[0],Paclon.shape[0]))+np.nan
    PacEOF[~np.isnan(PacificSST_flat[:,:,0])] = evec
    
    PDOindex = np.matmul(np.transpose(Pacnoland),evec)
    PDOindex = np.reshape(PDOindex,(100,252))
    
    da_PDO = xr.DataArray(
        data=PDOindex,
        dims=["variant","year"],
        coords=dict(
            year=year,
            variant=variant
        ),
        attrs=dict(
            description="PDOindex",
            units="covariance",
        ),
    )
    
    dashort = da_PDO.sel(year=slice(daterange[0],daterange[1]))
    
    return dashort
    
def SPGSST(daterange=[1980, 2100]):
    # load in SST annual means
    # path = "data/"
    ds = xr.open_dataset(path+"LENS_SSTannualmean_fullfield.nc")
    
    lat1 = 40
    lat2 = 60
    lon1 = 280
    lon2 = 340
    
    # grab region to calculate over
    SST = ds.SST
    SSTregion = SST.sel(year=slice(daterange[0],daterange[1]),lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    
    fr = SSTregion.mean(dim="variant")
    SST_IV = SSTregion-fr
    
    
    weights = np.cos(np.deg2rad(SST_IV.lat))
    SPG = SST_IV.weighted(
        weights).mean(dim=("lat", "lon"))

    return SPG


