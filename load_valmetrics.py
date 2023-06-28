#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:07:20 2023

@author: emilygordon
"""


# load in the models
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

import tensorflow as tf

#%%

ly1 = 1
ly2 = 5
averaginglength = 10

trendlength = 10

trainensnums = [0,1,2] # which pair from each set of 10
valensnums = [3]
testensnums = [4]

trainens = preprocessing.enslists(trainensnums)
valens = preprocessing.enslists(valensnums)
testens = preprocessing.enslists(testensnums)

experiment_name = "predictSST_terciles20202050_singlelocation_nobias_betterpreprocessing"
experiment_dict = experiment_settings.get_experiment_settings(experiment_name)
filefront = experiment_dict["filename"]

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


#%%

trainingdate1 = experiment_dict["trainingdate1"]
trainingdate2 = experiment_dict["trainingdate2"]
date1 = experiment_dict["cutoff1date"]
date2 = experiment_dict["cutoff2date"]
lcutoff_q = experiment_dict["lcutoff"]
ucutoff_q = experiment_dict["ucutoff"]
latres = experiment_dict["latres"]
lonres = experiment_dict["lonres"]

daterange = [trainingdate1,trainingdate2]

# validation
SST1a = preprocessing.SSTinput_nft(ly1,averaginglength,daterange,trendlength)
SST1b = preprocessing.SSTinput_nft(ly2,averaginglength,daterange,trendlength)
SST2 = preprocessing.SSTinput_ft(ly1,averaginglength,daterange,trendlength)

lon = np.asarray(SST2.lon)
lat = np.asarray(SST2.lat)

X1aval = preprocessing.memsplit(SST1a,valens)
X1bval = preprocessing.memsplit(SST1b,valens)
X2val = preprocessing.memsplit(SST2,valens)

X1val = np.concatenate((X1aval,X1bval),axis=1)

latvec = np.arange(-60,70,10)
lonvec = np.arange(0,360,10)

nlat = len(latvec)
nlon = len(lonvec)

ntestmems = len(testens)
xvec = np.arange(daterange[0],daterange[1]-trendlength+1)
xvecvec = np.tile(xvec,ntestmems)
boo2050 = (xvecvec>=2020) & (xvecvec<2050)

seeds = np.arange(70,75)
nseeds = len(seeds)
nconfs = 10

accval_2050 = np.empty((nlat,nlon,nseeds,nconfs))+np.nan
lossval_2050 = np.empty((nlat,nlon,nseeds))+np.nan

for ilat,lat1 in enumerate(latvec):
    lat2 = lat1+latres
    for ilon, lon1 in enumerate(lonvec):        
        lon2 = lon1+lonres
        
        print([lat1,lon1])
        
        SSTout=preprocessing.SSToutput(trendlength,lat1,lat2,lon1,lon2,daterange)
        if ~np.isnan(np.asarray(SSTout)[0,0]):

            ucutoff = preprocessing.calc_ucutoff_q(SSTout, trainens, date1, date2, ucutoff_q)
            lcutoff = preprocessing.calc_lcutoff_q(SSTout, trainens, date1, date2, lcutoff_q)
    
            Yval = preprocessing.memsplit(SSTout,valens)
            Yval = preprocessing.y_onehot(Yval,lcutoff,ucutoff)

            
            for iseed,seed in enumerate(seeds):
                
                model,_,_ = loadmodel(experiment_dict,X1val.shape[1],X2val.shape[1],lat1,lon1,seed)
                
                ypredval = model.predict({"deforced":X1val,
                                       "forced":X2val})
                
                accval_2050[ilat,ilon,iseed,:] = metrics.confxacc(ypredval[boo2050],Yval[boo2050])
                lossval_2050[ilat,ilon,iseed] = metrics.loss(ypredval[boo2050],Yval[boo2050])          

#%% save metrics

valfile = "metricsfiles/" + filefront + "val_2050.npz"
np.savez(valfile,
         acc=accval_2050,
         loss=lossval_2050)







