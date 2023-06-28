#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:13:34 2023

@author: emgordy
"""

# Figure 3a and S1

import xarray as xr
import numpy as np
import sys
import os
import gc

path=os.getcwd()
sys.path.append(path+'/functions/')
# custom modules, les miennes
import preprocessing
import metrics
import ANN
import experiment_settings
import weights_calcs
import glob

from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score,pairwise_distances,euclidean_distances
from scipy.stats import pearsonr

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors
import cmasher as cmr

#pretty plots
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi']= 150
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['font.size'] = 12

params = {"ytick.color" : "k",
          "xtick.color" : "k",
          "axes.labelcolor" : "k",
          "axes.edgecolor" : "k"}
plt.rcParams.update(params)

#%%

experiment_name = "predictSST_terciles20202050_singlelocation_nobias_betterpreprocessing"
experiment_dict = experiment_settings.get_experiment_settings(experiment_name)
filefront = experiment_dict["filename"]
trainingdate1 = experiment_dict["trainingdate1"]
trainingdate2 = experiment_dict["trainingdate2"]
date1 = experiment_dict["cutoff1date"]
date2 = experiment_dict["cutoff2date"]
lcutoff_q = experiment_dict["lcutoff"]
ucutoff_q = experiment_dict["ucutoff"]
trendlength=10
daterange = [trainingdate1,trainingdate2]

filename = "metricsfiles/" + filefront + "allpreds.npz"
bigboifile = np.load(filename)

projection = ccrs.EqualEarth(central_longitude=205)
transform = ccrs.PlateCarree()

file = "metricsfiles/" + filefront + "test_2050.npz"
test_2050=np.load(file)
acctest = test_2050["acc"]
acctest = acctest[:,:,0]
shuffle = test_2050['shuffleacc']
shuffle = shuffle[:,:,0]
shuffle90 = test_2050["shuffle90"]
shuffle90 = shuffle90[:,:,0]
seedfile = "metricsfiles/" + filefront + "bestseeds.npz"
seedrankfile=np.load(seedfile)
seedrank=seedrankfile["seedrank"]
nseedsel = 3
seedchoose = seedrank[:,:,:nseedsel]

#%%
mask1 = shuffle90[:,:,0]>=acctest[:,:,0]

lon = np.arange(0,360,10,dtype=float)
lat = np.arange(-60,70,10,dtype=float)

longrid,latgrid = np.meshgrid(lon+5,lat+5)

ilons = np.arange(36)
ilats = np.arange(13)

lonnan = longrid[mask1]
latnan = latgrid[mask1]

longrid[mask1] = np.nan
latgrid[mask1] = np.nan

longrid[np.isnan(shuffle90[:,:,0])] = np.nan
latgrid[np.isnan(shuffle90[:,:,0])] = np.nan

lonsel = [190,330,270,60]
latsel = [20,60,-50,-30]

nsel = 4

plt.figure(figsize=(8,5))
a0=plt.subplot(1,1,1,projection=projection)
a0.scatter(longrid,latgrid,transform=transform,marker='o',s=50,color='xkcd:golden rod')
a0.scatter(lonnan,latnan,transform=transform,marker='x',color='grey')
a0.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))

#%%

testensnums = [4]
testens = preprocessing.enslists(testensnums)

ntestmems = len(testens)
xvec = np.arange(daterange[0],daterange[1]-trendlength+1)
xvecvec = np.tile(xvec,ntestmems)
boo2050 = (xvecvec>=2020) & (xvecvec<2050)

bigacc = bigboifile["acc"]
bigacc = bigacc[:,:,0,boo2050]

xvec = np.arange(600)
nplot = 60

for i in range(nsel):
    
    ilonsel = np.argwhere(lon==lonsel[i])[0][0]
    ilatsel = np.argwhere(lat==latsel[i])[0][0]
    
    acctseries = bigacc[ilatsel,ilonsel]
    
    plt.figure(figsize=(6,2))
    plt.plot(xvec[:nplot],acctseries[:nplot],'o-',color='xkcd:burnt orange')
    plt.yticks([0,1])
    plt.xticks(np.arange(0,nplot+10,10),labels=[])
    plt.show()


#%%

def loadmodel(experiment_dict,train1shape,train2shape,lat1,lon1,seed):
    
    dropout_rate = experiment_dict["dropout_rate"]
    ridge_param1 = experiment_dict["ridge_param"][0]
    ridge_param2 = experiment_dict["ridge_param"][1]
    hiddens1 = experiment_dict["hiddens1"]
    hiddens2 = experiment_dict["hiddens2"]
    model1name = experiment_dict["model1name"]
    model2name = experiment_dict["model2name"]
    filefront = experiment_dict["filename"]
    latres = experiment_dict["latres"]
    lonres = experiment_dict["lonres"]
    
    input_shape1 = train1shape
    input_shape2 = train2shape

    lat2 = lat1+latres   
    lon2 = lon1+lonres
    print([lat1,lon1])

    modelstr = ANN.multimodelstr(filefront,lat1,lat2,lon1,lon2,seed) 
    modelcheck = glob.glob(modelstr)
    if len(modelcheck) != 0:
        print(modelstr)
        tf.random.set_seed(seed)
        np.random.seed(seed)     
        x1,input1 = ANN.build_model_api_nobias(seed,dropout_rate,ridge_param1,hiddens1,input_shape1,model1name)
        x2,input2 = ANN.build_model_api(seed+1,dropout_rate,ridge_param2,hiddens2,input_shape2,model2name)   
        model = ANN.fullmodel(x1,x2,input1,input2,seed)


        model.load_weights(modelstr)
        
        return model,x1,x2
    else:
        print("model does not exist")


filefront = experiment_dict["filename"]
trainingdate1 = experiment_dict["trainingdate1"]
trainingdate2 = experiment_dict["trainingdate2"]
date1 = experiment_dict["cutoff1date"]
date2 = experiment_dict["cutoff2date"]
lcutoff_q = experiment_dict["lcutoff"]
ucutoff_q = experiment_dict["ucutoff"]

file2 = "metricsfiles/"+filefront+"val_2050.npz"

val_2050=np.load(file2)
loss_2050=val_2050["loss"]

nseedsel = 3
seedrank = np.argsort(loss_2050,axis=2)
seedchoose = seedrank[:,:,:nseedsel]

# prepare data

ly1 = 1
ly2 = 5
averaginglength = 10
trendlength = 10
trainensnums = [0,1,2] # which pair from each set of 10
trainens = preprocessing.enslists(trainensnums)
testensnums = [4]
testens = preprocessing.enslists(testensnums)

daterange = [trainingdate1,trainingdate2]

SST1a = preprocessing.SSTinput_nft(ly1,averaginglength,daterange,trendlength)
SST1b = preprocessing.SSTinput_nft(ly2,averaginglength,daterange,trendlength)
SST2 = preprocessing.SSTinput_ft(ly1,averaginglength,daterange,trendlength)

# input data
X1atest = preprocessing.memsplit(SST1a,testens)
X1btest = preprocessing.memsplit(SST1b,testens)
X1test = np.concatenate((X1atest,X1btest),axis=1)
X2test = preprocessing.memsplit(SST2,testens)

# select region and seed
latvec = np.arange(-60,70,10)
lonvec = np.arange(0,360,10)

lat1 = latsel[3]
lon1 = lonsel[3]

lat2 = lat1+10
lon2 = lon1+10

ilat = np.argwhere(latvec==lat1)[0][0]
ilon = np.argwhere(lonvec==lon1)[0][0]
seed = seedchoose[ilat,ilon,0]+70

# output data
SSTout = preprocessing.SSToutput(trendlength,lat1,lat2,lon1,lon2,daterange)

ucutoff = preprocessing.calc_ucutoff_q(SSTout, trainens, date1, date2, ucutoff_q)
lcutoff = preprocessing.calc_lcutoff_q(SSTout, trainens, date1, date2, lcutoff_q)

Ytest = preprocessing.memsplit(SSTout,testens)
Ytest = preprocessing.y_onehot(Ytest,lcutoff,ucutoff)

ntestmems = len(testens)
xvec = np.arange(daterange[0],daterange[1]-trendlength+1)
xvecvec = np.tile(xvec,ntestmems)
boo2050 = (xvecvec>=2020) & (xvecvec<2050)

label = np.argmax(Ytest,axis=1)
label = label[boo2050]

allx = np.tile(np.arange(2020,2050),20)
allens = np.tile(testens,(30,1))
allens = np.reshape(allens,(600),'F')

#%%

model,x1,x2 = loadmodel(experiment_dict,X1test.shape[1],X2test.shape[1],lat1,lon1,seed)
ypredtest = model.predict({"deforced":X1test[boo2050,:],
                          "forced":X2test[boo2050,:]})

pred = np.argmax(ypredtest,axis=1)

IVnodes,EFnodes = weights_calcs.getoutputvecs(model,x1,x2,X1test[boo2050,:],X2test[boo2050,:])

IVwin = np.argmax(IVnodes,axis=1)
EFwin = np.argmax(EFnodes,axis=1)

IVcorr = IVwin==label
EFcorr = EFwin==label
modelcorr = pred==label

effectofIV1 = IVcorr & modelcorr
effectofIV2 = ~EFcorr & ~IVcorr & modelcorr

#%% supplement figure schematic a.

X1plot = X1test[boo2050]
X1plot = X1plot[effectofIV1]
X1plot = X1plot[14,:]

X2plot = X2test[boo2050]
X2plot = X2plot[effectofIV1]
X2plot = X2plot[14,:]

X1aplot = np.reshape(X1plot[:36*72],(36,72))
X1bplot = np.reshape(X1plot[36*72:],(36,72))
X2plot = np.reshape(X2plot,(36,72))

lonplot = np.arange(2.5,360,5)
latplot = np.arange(-87.5,90,5)

projection = ccrs.PlateCarree(central_longitude=180)

plt.figure(figsize=(8,5))
a0=plt.subplot(1,1,1,projection=projection)
a0.pcolormesh(lonplot,latplot,X1aplot,transform=transform,cmap="RdBu_r",vmin=-0.8,vmax=0.8)
a0.coastlines(color='gray')
plt.show()

plt.figure(figsize=(8,5))
a0=plt.subplot(1,1,1,projection=projection)
a0.pcolormesh(lonplot,latplot,X1bplot,transform=transform,cmap="RdBu_r",vmin=-0.8,vmax=0.8)
a0.coastlines(color='gray')
plt.show()

plt.figure(figsize=(8,5))
a0=plt.subplot(1,1,1,projection=projection)
a0.pcolormesh(lonplot,latplot,X2plot,transform=transform,cmap="RdBu_r",vmin=-1.2,vmax=1.2)
a0.coastlines(color='gray')
plt.show()

IVnodesout = IVnodes[effectofIV1]
IVnodesout = IVnodesout[14,:]

EFnodesout = EFnodes[effectofIV1]
EFnodesout = EFnodesout[14,:]

allnodesout = ypredtest[effectofIV1]
allnodesout = allnodesout[14,:]

print(IVnodesout)
print(EFnodesout)
print(allnodesout)

mem = allens[effectofIV1][14]
yearout = allx[effectofIV1][14]

print(mem)
print(yearout)

#%% supplement figure schematic b.

X1plot = X1test[boo2050]
X1plot = X1plot[effectofIV1]
X1plot = X1plot[8,:]

X2plot = X2test[boo2050]
X2plot = X2plot[effectofIV1]
X2plot = X2plot[8,:]

X1aplot = np.reshape(X1plot[:36*72],(36,72))
X1bplot = np.reshape(X1plot[36*72:],(36,72))
X2plot = np.reshape(X2plot,(36,72))

lonplot = np.arange(2.5,360,5)
latplot = np.arange(-87.5,90,5)

projection = ccrs.PlateCarree(central_longitude=180)

plt.figure(figsize=(8,5))
a0=plt.subplot(1,1,1,projection=projection)
a0.pcolormesh(lonplot,latplot,X1aplot,transform=transform,cmap="RdBu_r",vmin=-0.8,vmax=0.8)
a0.coastlines(color='gray')
plt.show()

plt.figure(figsize=(8,5))
a0=plt.subplot(1,1,1,projection=projection)
a0.pcolormesh(lonplot,latplot,X1bplot,transform=transform,cmap="RdBu_r",vmin=-0.8,vmax=0.8)
a0.coastlines(color='gray')
plt.show()

plt.figure(figsize=(8,5))
a0=plt.subplot(1,1,1,projection=projection)
a0.pcolormesh(lonplot,latplot,X2plot,transform=transform,cmap="RdBu_r",vmin=-1.2,vmax=1.2)
a0.coastlines(color='gray')
plt.show()

IVnodesout = IVnodes[effectofIV1]
IVnodesout = IVnodesout[8,:]

EFnodesout = EFnodes[effectofIV1]
EFnodesout = EFnodesout[8,:]

allnodesout = ypredtest[effectofIV1]
allnodesout = allnodesout[8,:]

print(IVnodesout)
print(EFnodesout)
print(allnodesout)

mem = allens[effectofIV1][8]
yearout = allx[effectofIV1][8]

print(mem)
print(yearout)

#%% supplement figure schematic c.

X1plot = X1test[boo2050]
X1plot = X1plot[effectofIV2]
X1plot = X1plot[3,:]

X2plot = X2test[boo2050]
X2plot = X2plot[effectofIV2]
X2plot = X2plot[3,:]

X1aplot = np.reshape(X1plot[:36*72],(36,72))
X1bplot = np.reshape(X1plot[36*72:],(36,72))
X2plot = np.reshape(X2plot,(36,72))

lonplot = np.arange(2.5,360,5)
latplot = np.arange(-87.5,90,5)

projection = ccrs.PlateCarree(central_longitude=180)

plt.figure(figsize=(8,5))
a0=plt.subplot(1,1,1,projection=projection)
a0.pcolormesh(lonplot,latplot,X1aplot,transform=transform,cmap="RdBu_r",vmin=-0.8,vmax=0.8)
a0.coastlines(color='gray')
plt.show()

plt.figure(figsize=(8,5))
a0=plt.subplot(1,1,1,projection=projection)
a0.pcolormesh(lonplot,latplot,X1bplot,transform=transform,cmap="RdBu_r",vmin=-0.8,vmax=0.8)
a0.coastlines(color='gray')
plt.show()

plt.figure(figsize=(8,5))
a0=plt.subplot(1,1,1,projection=projection)
a0.pcolormesh(lonplot,latplot,X2plot,transform=transform,cmap="RdBu_r",vmin=-1.2,vmax=1.2)
a0.coastlines(color='gray')
plt.show()

IVnodesout = IVnodes[effectofIV2]
IVnodesout = IVnodesout[3,:]

EFnodesout = EFnodes[effectofIV2]
EFnodesout = EFnodesout[3,:]

allnodesout = ypredtest[effectofIV2]
allnodesout = allnodesout[3,:]

print(IVnodesout)
print(EFnodesout)
print(allnodesout)

mem = allens[effectofIV2][3]
yearout = allx[effectofIV2][3]

print(mem)
print(yearout)











