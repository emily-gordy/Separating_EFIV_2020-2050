#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:20:12 2023

@author: emgordy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys


sys.path.append('functions/')

# custom modules, les miennes
import experiment_settings
import metricplots
import importlib as imp
imp.reload(metricplots)


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

experiment_name = "predictSST_terciles20202050_singlelocation_nobias_betterpreprocessing"

# experiment_name = "predictT2M_terciles20202050_singlelocation"
experiment_dict = experiment_settings.get_experiment_settings(experiment_name)
filefront = experiment_dict["filename"]

#%%
lat = np.arange(-60,70,10)+5
lon = np.arange(0,360,10)+5

nlat = len(lat)
nlon = len(lon)

#%%

file = "metricsfiles/" + filefront + "test_2050.npz"
test_2050=np.load(file)

#%%

acc = test_2050["acc"]
acc = acc[:,:,0,:]

shuffle = test_2050["shuffleacc"]
shuffle = shuffle[:,:,0,:]


shuffle90 = test_2050["shuffle90"]
shuffle90 = shuffle90[:,:,0,:]

metricplots.compareaccuracy_IV_sig(acc,shuffle,lon,lat,shuffle90)
metricplots.compareaccuracy_IVSDP50_sig(acc,shuffle,lon,lat,shuffle90)
metricplots.compareaccuracy_IVSDP_sig(acc,shuffle,lon,lat,shuffle90)
metricplots.compareaccuracy_SDP_allish(acc,lon,lat)



