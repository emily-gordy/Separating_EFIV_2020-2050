#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:27:25 2023

@author: emgordy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

sys.path.append('functions/')
# custom modules, les miennes
import experiment_settings
import preprocessing


mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['figure.dpi']= 200
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['font.size'] = 12

params = {"ytick.color" : "k",
          "xtick.color" : "k",
          "axes.labelcolor" : "k",
          "axes.edgecolor" : "k"}
plt.rcParams.update(params)

#%% experiment parameters

ly1 = 1
ly2 = 5
averaginglength = 10
trendlength = 10
trainensnums = [0,1,2] # which pair from each set of 10
trainens = preprocessing.enslists(trainensnums)

def plottimeseries(lat1,lon1,enssel,experiment_name):
    
    experiment_dict = experiment_settings.get_experiment_settings(experiment_name)
    trainingdate1 = experiment_dict["trainingdate1"]
    trainingdate2 = experiment_dict["trainingdate2"]
    date1 = experiment_dict["cutoff1date"]
    date2 = experiment_dict["cutoff2date"]
    lcutoff_q = experiment_dict["lcutoff"]
    ucutoff_q = experiment_dict["ucutoff"]

    # prepare data
    
    daterange = [trainingdate1,trainingdate2]
    
    lat2 = lat1+10
    lon2 = lon1+10
    
    # output data
    SSTout = preprocessing.SSToutput(trendlength,lat1,lat2,lon1,lon2,daterange)
    
    ucutoff = preprocessing.calc_ucutoff_q(SSTout, trainens, date1, date2, ucutoff_q)
    lcutoff = preprocessing.calc_lcutoff_q(SSTout, trainens, date1, date2, lcutoff_q)
    
    # everything else
    SSTtrendvec = np.asarray(SSTout[:,enssel])
    SSTregionmean,year,yintvec = preprocessing.SSToutput_all(trendlength,lat1,lat2,lon1,lon2,daterange)
    yintvec = np.asarray(yintvec)
    
    xvec = np.arange(daterange[0],daterange[1]-trendlength+1)
    boo2050 = (xvec>=2020) & (xvec<2050)
    SSTtrend2050 = np.asarray(SSTout[boo2050,:]).flatten()
    
    tvec = np.arange(0,1,1/trendlength)
    timeshort = np.asarray(year.sel(year=slice(daterange[0],daterange[1]-trendlength)))
    npoints = len(timeshort)    
    timefull = np.asarray(year.sel(year=slice(daterange[0],daterange[1])))

    # plotting
    memsel = enssel[0]
    
    plt.figure(figsize=(5,4))
    plt.plot(timefull,np.transpose(SSTregionmean),color='xkcd:grey',linewidth=0.4,alpha=0.8)
    plt.plot(timefull,SSTregionmean[memsel,:],color='xkcd:grey',linewidth=0.4,alpha=0.8,label='all members')
    plt.plot(timefull,np.mean(SSTregionmean,axis=0),color='xkcd:black',linewidth=1.5,label='forced response')
    plt.plot(timefull,SSTregionmean[memsel,:],color='xkcd:royal blue',linewidth=0.8,label='single member')
    plt.ylabel(r'SST ($^{\circ}$C)')
    plt.xlabel('year')
    # plt.title('a. All trends')
    
    iline = np.arange(0,npoints,10)
    for ind in iline:
        timeloop = timefull[ind:ind+trendlength]
        yloop = SSTtrendvec[ind]*tvec+yintvec[ind,memsel]
        if SSTtrendvec[ind]<lcutoff:
            linecolor="xkcd:teal"
        elif SSTtrendvec[ind]>=ucutoff:
            linecolor='xkcd:red'
        else:
            linecolor="xkcd:orange"
        
        plt.plot(timeloop,yloop,color=linecolor,linewidth=1.8)
    
    plt.xlim(1960,2100)
    # plt.ylim(12,15.5)
    plt.xticks(np.arange(1960,2120,20))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(5,4))
    plt.hist(SSTtrend2050[SSTtrend2050<=lcutoff],bins=np.arange(-1.5,2.1,0.1),color='xkcd:teal',label='lower tercile')
    plt.hist(SSTtrend2050[(SSTtrend2050>lcutoff) & (SSTtrend2050<=ucutoff)],bins=np.arange(-1.5,2.1,0.1),color='xkcd:orange',label='middle tercile')
    plt.hist(SSTtrend2050[SSTtrend2050>ucutoff],bins=np.arange(-1.5,2.1,0.1),color='xkcd:red',label='upper tercile')
    plt.ylabel('count')
    plt.xlabel(r'SST trend ($^{\circ}$C/decade)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    

#%%
lat1 = -50
lon1 = 240
enssel = [69]
experiment_name = "predictSST_terciles20202050_singlelocation_nobias_betterpreprocessing"    
plottimeseries(lat1,lon1,enssel,experiment_name)   
    
    
    
    
    
    
    
    
    
    
    
    
    