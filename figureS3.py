#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:06:32 2023

@author: emgordy
"""

# regional plotting
# with regions decided by clustering
# and then plot the cluster
# silhouette score

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
kmeans = KMeans(n_clusters=nclusters,n_init=400,random_state=28)

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

kclustermatIV_small+= 1

# kclustermatIV_small[kclustermatIV_small==5]=100
# kclustermatIV_small[kclustermatIV_small==2]=5
# kclustermatIV_small[kclustermatIV_small==100]=2

# kclustermatIV_small[kclustermatIV_small==5]=100
# kclustermatIV_small[kclustermatIV_small==4]=5
# kclustermatIV_small[kclustermatIV_small==100]=4


kclustermatIV_small[np.isnan(kclustermatIV_small)] = 0

k_range = range(2, 10)
# randomstates = range(29,29)

# Initialize list to store silhouette scores
silhouette_scores = []
wcss = []

# Test different number of clusters
for k in k_range:
    silhouettevec = []
        # Create KMeans model
    kmeans = KMeans(n_clusters=k,n_init=400,random_state=29)
    
    # Fit the model to the data
    kmeans.fit(IVaccsmall)
    
    # Get cluster labels
    labels = kmeans.labels_
    wcss.append(kmeans.inertia_)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(IVaccsmall, labels)
    # Append score to list
    silhouette_scores.append(silhouette_avg)

#%%
silhouette_scores = np.asarray(silhouette_scores)

plt.figure(figsize=(7,3))
ax1 = plt.subplot(1,1,1)
# plt.plot(k_range,silhouette_scores,color='xkcd:light gray',linewidth='0.8')
# plt.plot(k_range,silhouette_scores[:,0],color='xkcd:light gray',linewidth='0.8',label='single random state')
ax1.plot(k_range,silhouette_scores,color='xkcd:dark teal',label='silhouette score')
ax1.set_xlabel('n clusters')
ax1.set_ylabel('silhouette score')

# ax2 = ax1.twinx()
# ax2.plot(k_range,wcss,color='xkcd:indian red',label='wcss')
# ax2.set_ylabel('within cluster sum of square')

plt.tight_layout()
plt.show()
