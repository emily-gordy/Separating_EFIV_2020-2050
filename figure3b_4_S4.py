 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:54:43 2023

@author: emgordy
"""

# regional plotting
# with regions decided by clustering
# and then plot the cluster

import numpy as np
import sys
import os

path=os.getcwd()
sys.path.append(path+'/functions/')
# custom modules, les miennes
import preprocessing
import experiment_settings

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,euclidean_distances
from scipy.stats import pearsonr

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

#%% all experiment parameters

experiment_name = "predictSST_terciles20202050_singlelocation_nobias_betterpreprocessing"
experiment_dict = experiment_settings.get_experiment_settings(experiment_name)
filefront = experiment_dict["filename"]
trainingdate1 = experiment_dict["trainingdate1"]
trainingdate2 = experiment_dict["trainingdate2"]
date1 = experiment_dict["cutoff1date"]
date2 = experiment_dict["cutoff2date"]
lcutoff_q = experiment_dict["lcutoff"]
ucutoff_q = experiment_dict["ucutoff"]

ly1 = 1
ly2 = 5
averaginglength = 10
trendlength = 10
trainensnums = [0,1,2] # which pair from each set of 10
trainens = preprocessing.enslists(trainensnums)

testensnums = [4]
testens = preprocessing.enslists(testensnums)

trainingdate1 = experiment_dict["trainingdate1"]
trainingdate2 = experiment_dict["trainingdate2"]

daterange = [trainingdate1,trainingdate2]

ntestmems = len(testens)
xvec = np.arange(daterange[0],daterange[1]-trendlength+1)
xvecvec = np.tile(xvec,ntestmems)
boo2050 = (xvecvec>=2020) & (xvecvec<2050)


#%% experiment metrics
filename = "metricsfiles/" + filefront + "allpreds.npz"
bigboifile = np.load(filename)

file = "metricsfiles/" + filefront + "test_2050.npz"
test_2050=np.load(file)
acctest = test_2050["acc"]
acctest = acctest[:,:,0]
shuffle90 = test_2050["shuffle90"]
shuffle90 = shuffle90[:,:,0]

mask1 = shuffle90[:,:,0]>=acctest[:,:,0]

IVacc = bigboifile["IVacc"]

IVacc = IVacc[:,:,0,boo2050]
IVacc_mask = np.copy(IVacc)
IVacc_mask[mask1,:]=np.nan
IVaccsmall=IVacc_mask[~np.isnan(IVacc_mask[:,:,0]),:]

IVaccsmall_boo = IVaccsmall==1

#%% Kmeans using scikit learn

nclusters = 6

# Create KMeans model
kmeans = KMeans(n_clusters=nclusters,n_init=400,random_state=29)

# Fit the model to the data
kmeans.fit(IVaccsmall)

# Get cluster labels
labels = kmeans.labels_
kclustermatIV_small = np.empty((13,36))+np.nan
kclustermatIV_small[~np.isnan(IVacc_mask[:,:,0])]=labels

centroids = kmeans.cluster_centers_

euc_res = euclidean_distances(centroids,IVaccsmall)
distfromcent = np.min(euc_res,axis=0)

labelind = np.arange(len(labels))

corrs = np.empty(len(labels))

for icent in range(nclusters):
    
    centloop = centroids[icent,:]
    labelloop = labelind[labels==icent]
    
    for isamp,ind in enumerate(labelloop):
        corrs[ind],_ = pearsonr(centloop,IVaccsmall[ind,:])

kclustermatdist = np.empty((13,36))+np.nan
kclustermatdist[~np.isnan(IVacc_mask[:,:,0])]=distfromcent

kclustermatcorr = np.empty((13,36))+np.nan
kclustermatcorr[~np.isnan(IVacc_mask[:,:,0])]=corrs

kclustermatIV_small[kclustermatIV_small==5]=100
kclustermatIV_small[kclustermatIV_small==2]=5
kclustermatIV_small[kclustermatIV_small==100]=2

kclustermatIV_small[kclustermatIV_small==5]=100
kclustermatIV_small[kclustermatIV_small==4]=5
kclustermatIV_small[kclustermatIV_small==100]=4

kclustermatIV_small+= 1

kclustermatIV_small[np.isnan(kclustermatIV_small)] = 0

#%% plot cluster also dist from centroid and also correlation with centroid

lon = np.arange(0,360,10)
lat = np.arange(-60,70,10)

longrid,latgrid = np.meshgrid(lon+5,lat+5)
siglon = longrid[acctest[:,:,0]<=shuffle90[:,:,0]]
siglat = latgrid[acctest[:,:,0]<=shuffle90[:,:,0]]

projection = ccrs.EqualEarth(central_longitude=205)
transform = ccrs.PlateCarree()

cmap_cluster = mpl.colormaps['inferno']
cmapint = np.asarray(cmap_cluster.colors)
cmapout = np.concatenate((cmapint[1,np.newaxis],cmapint[30:]),axis=0)
cmapnew = mpl.colors.LinearSegmentedColormap.from_list('cmap_name', cmapout) # brightening the colormap for clustering

vmin = -0.5
vmax = 7.5

bounds = np.arange(vmin,vmax,1)
norm_clust = colors.BoundaryNorm(boundaries=bounds, ncolors=cmap_cluster.N)

plt.figure(figsize=(12,11))

a0=plt.subplot(3,1,1,projection=projection)
c0=a0.pcolormesh(lon+5,lat+5,kclustermatIV_small,transform=transform,cmap=cmapnew,norm=norm_clust)
a0.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
cbar0=plt.colorbar(c0,ticks=np.arange(0,7))
cbar0.ax.set_yticklabels(['NA','1','2','3','4','5','6'])
cbar0.ax.set_ylabel('n cluster')

a1=plt.subplot(3,1,2,projection=projection)
c1=a1.pcolormesh(lon+5,lat+5,kclustermatdist,transform=transform,cmap='magma_r')
a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
cbar1=plt.colorbar(c1)
cbar1.ax.set_ylabel('dist from centroid')

a1=plt.subplot(3,1,3,projection=projection)
c1=a1.pcolormesh(lon+5,lat+5,kclustermatcorr,transform=transform,cmap='magma')
a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
cbar1=plt.colorbar(c1)
cbar1.ax.set_ylabel('correlation with centroid')

plt.tight_layout()
plt.show() 


#%%

def plotclusters(kclustermatIV_small,clustersel):

    lonssel = loninds[kclustermatIV_small==clustersel]
    latssel = latinds[kclustermatIV_small==clustersel]
    
    nmodels = len(lonssel)    
    
    lonmodels = longrid[kclustermatIV_small==clustersel]
    latmodels = latgrid[kclustermatIV_small==clustersel]
    
    accregion = accall[latssel,lonssel,:,:]
    IVaccregion = IVacc[latssel,lonssel,:,:]
    EFaccregion = EFacc[latssel,lonssel,:,:]
    labelregion = label[latssel,lonssel,:,:]
    
    accregion = accregion[:,0,boo2050]
    IVaccregion = IVaccregion[:,0,boo2050]
    EFaccregion = EFaccregion[:,0,boo2050]
    labelregion = labelregion[:,0,boo2050]
    
    effectofIV = ((accregion==1) & (IVaccregion==1))

    effectofIVmean = np.nansum(effectofIV,axis=0)
    # if clustersel == 5:
    #     labelregionmean = np.mean(labelregion[-5:,:],axis=0)
    # elif clustersel == 4:
    #     labelregionmean = np.mean(labelregion[1:13,:],axis=0)
    # elif clustersel == 3:
    #     labelregionmean = np.mean(labelregion[3:-3],axis=0)
    labelregionmean = np.mean(labelregion,axis=0)
    labelregion_all = np.empty((len(effectofIVmean),3))
    
    for i in range(3):
        labelregion_all[:,i] = np.mean(1*(labelregion==i),axis=0)
    
    cutoff = (1/2)*nmodels
    
    xvec = np.arange(600)
    IV_scatter = effectofIVmean[effectofIVmean>cutoff]
    xvec_scatter = xvec[effectofIVmean>cutoff]
    labelregion_scatter = labelregion_all[effectofIVmean>cutoff,:]
    
    predlists = [labelregionmean>1,
                 labelregionmean<1]
    
    plt.figure(figsize=(10,12))
    
    plt.subplot(4,1,1)
    plt.plot(xvec,effectofIVmean,color='xkcd:forest green',zorder=0)
    plt.scatter(xvec_scatter,IV_scatter,color='xkcd:golden rod',zorder=1)
    plt.ylabel('n models correct (%d total)' %(nmodels))
    plt.xlim(0,600)
    
    
    for iplot in range(3):
        plt.subplot(4,1,iplot+2)
        plt.plot(xvec,labelregion_all[:,iplot],color=labelcolors[iplot],zorder=0)
        plt.scatter(xvec_scatter,labelregion_scatter[:,iplot],color='xkcd:golden rod',zorder=1)
        plt.xlim(0,600)
    plt.xlabel('testing sample timeseries')
    # plt.ylabel('avg label value (0 = all low, 2 = all high)')
    
    
    plt.tight_layout()
    plt.show()
    
    if len(IV_scatter>0):
    
        for ipred,predsel in enumerate(predlists):    
        
            X1_sel = X1_2050[(predsel & (effectofIVmean>=cutoff)),:]
            
            nsamps = X1_sel.shape[0]
            print(nsamps)
            
            mapplot1 = np.reshape(np.mean(X1_sel[:,:2592],axis=0),(36,72))
            mapplot1,lonp = add_cyclic_point(mapplot1,coord=lon) 
            
            mapplot2 = np.reshape(np.mean(X1_sel[:,2592:],axis=0),(36,72))
            mapplot2,_ = add_cyclic_point(mapplot2,coord=lon)
        
            plt.figure(figsize=(10,8))
        
            a0=plt.subplot(2,1,2,projection=projection)
            a0.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
            a0.contourf(lonp,lat,mapplot1,sstcrange,cmap=sstcmap,transform=transform,extend='both')
            a0.scatter(lonmodels,latmodels,color='xkcd:forest green',transform=transform,facecolors='none',marker='s',s=100,zorder=12)
            a0.coastlines(color='gray')

        
            a1=plt.subplot(2,1,1,projection=projection)
            a1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='grey'))
            c2=a1.contourf(lonp,lat,mapplot2,sstcrange,cmap=sstcmap,transform=transform,extend='both')
            a1.scatter(lonmodels,latmodels,color='xkcd:forest green',transform=transform,facecolors='none',marker='s',s=100,zorder=12)

            plt.tight_layout()
            plt.show()
        
            predmask = (predsel & (effectofIVmean>=cutoff))
            
            if clustersel == 4:
                ninocorr = np.mean(nino2050[predmask,:],axis=0)
                plt.figure(figsize=(5,3))
                plt.plot(np.arange(-10,10),ninocorr,color='xkcd:burnt orange',linewidth=1.5,zorder=1,label="mean")
                plt.plot(np.arange(-10,10),np.transpose(nino2050[predmask,:]),color='xkcd:golden rod',linewidth=0.7,zorder=0)
                plt.xlim(-10,9)
                plt.vlines(0,np.min(nino2050[predmask,:]),np.max(nino2050[predmask,:]),color='grey',zorder=-1)
                plt.xlabel('lead year')
                plt.xticks(np.arange(-10,10,2))
                plt.ylabel('Annual Mean Nino 3.4')
                
                plt.tight_layout()
                plt.show()
            
            if clustersel == 5:
                
                PDOcorr = np.mean(PDO2050[predmask,:],axis=0)
                plt.figure(figsize=(5,3))
                plt.plot(np.arange(-10,10),PDOcorr,color='xkcd:burnt orange',linewidth=1.5,zorder=1,label='mean')
                plt.plot(np.arange(-10,10),np.transpose(PDO2050[predmask,:]),color='xkcd:golden rod',linewidth=0.7,zorder=0)
                plt.xlim(-10,9)
                plt.vlines(0,np.min(PDO2050[predmask,:]),np.max(PDO2050[predmask,:]),color='grey',zorder=-1)
                plt.xticks(np.arange(-10,10,2))
                plt.xlabel('lead year')
                plt.ylabel('Annual Mean PDO Index')
                
                plt.tight_layout()
                plt.show()
            
            if clustersel == 6:
                
                SPGcorr = np.nanmean(SPG2050[predmask,:],axis=0)
                plt.figure(figsize=(5,3))
                plt.plot(np.arange(-10,10),SPGcorr,color='xkcd:burnt orange',linewidth=1.5,zorder=1,label='mean')
                plt.plot(np.arange(-10,10),np.transpose(SPG2050[predmask,:]),color='xkcd:golden rod',linewidth=0.7,zorder=0)
                plt.plot(np.arange(-10,10),np.transpose(SPG2050[predmask,:])[:,0],color='xkcd:golden rod',linewidth=0.7,zorder=0,label='individual')
                plt.xlim(-10,9)
                plt.vlines(0,np.min(SPG2050[predmask,:]),np.max(SPG2050[predmask,:]),color='grey',zorder=-1)
                plt.xticks(np.arange(-10,10,2))
                plt.xlabel('lead year')
                plt.ylabel('Annual Mean NASPG')
                if ipred == 1:
                    plt.legend(loc='lower left')
                
                plt.tight_layout()
                plt.show()


#%%

SST1a = preprocessing.SSTinput_nft(ly1,averaginglength,daterange,trendlength)
SST1b = preprocessing.SSTinput_nft(ly2,averaginglength,daterange,trendlength)
SST2 = preprocessing.SSTinput_ft(ly1,averaginglength,daterange,trendlength)

daterange1 = [daterange[0]-10,daterange[1]]

nino34 = preprocessing.Nino34(daterange1)
nino34 = np.asarray(nino34)
ninoevmat = np.asarray([nino34[:,i:i+20] for i in range(131)])
ninotest = preprocessing.memsplit(ninoevmat,testens)

PDOindex = preprocessing.PDO(daterange1)
PDOindex = np.asarray(PDOindex)
PDOevmat = np.asarray([PDOindex[:,i:i+20] for i in range(131)])
PDOtest = preprocessing.memsplit(PDOevmat,testens)

SPGindex = preprocessing.SPGSST(daterange1)
SPGindex = np.asarray(SPGindex)
SPGevmat = np.asarray([SPGindex[:,i:i+20] for i in range(131)])
SPGtest = preprocessing.memsplit(SPGevmat,testens)

# input data
X1atest = preprocessing.memsplit(SST1a,testens)
X1btest = preprocessing.memsplit(SST1b,testens)
X1test = np.concatenate((X1atest,X1btest),axis=1)
X2test = preprocessing.memsplit(SST2,testens)
X1_2050 = X1test[boo2050,:]
X2_2050 = X2test[boo2050,:]

ninotest = preprocessing.memsplit(ninoevmat,testens)

nino2050 = ninotest[boo2050,:]
PDO2050 = PDOtest[boo2050,:]
SPG2050 = SPGtest[boo2050,:]

sstcmap = cmr.fusion_r
lrpcmap = cmr.redshift

ilons = np.arange(36)
ilats = np.arange(13)

loninds,latinds = np.meshgrid(ilons,ilats)

accall = bigboifile["acc"]
IVacc = bigboifile["IVacc"]
EFacc = bigboifile["EFacc"]
label = bigboifile["label"]

labelcolors = ["xkcd:teal","xkcd:orange","xkcd:red"]
weightstr = ['lower','middle','upper']

lon = np.arange(2.5,362.5,5)
lat = np.arange(-87.5,92.5,5)

sstcrange = np.arange(-0.6,0.62,0.02)


plotclusters(kclustermatIV_small,4)
plotclusters(kclustermatIV_small,5)
plotclusters(kclustermatIV_small,6)



