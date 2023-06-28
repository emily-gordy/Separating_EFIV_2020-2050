#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:42:42 2023

@author: emilygordon
"""

# load in the best 3 models at each location and calculate metrics on *testing data*

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
from scipy.stats import percentileofscore

import tensorflow as tf

#%% experiment parameters

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

#%% function to load in the model

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


#%% preprocessing

trainingdate1 = experiment_dict["trainingdate1"]
trainingdate2 = experiment_dict["trainingdate2"]
date1 = experiment_dict["cutoff1date"]
date2 = experiment_dict["cutoff2date"]
lcutoff_q = experiment_dict["lcutoff"]
ucutoff_q = experiment_dict["ucutoff"]
latres = experiment_dict["latres"]
lonres = experiment_dict["lonres"]

daterange = [trainingdate1,trainingdate2]

# testing
SST1a = preprocessing.SSTinput_nft(ly1,averaginglength,daterange,trendlength)
SST1b = preprocessing.SSTinput_nft(ly2,averaginglength,daterange,trendlength)
SST2 = preprocessing.SSTinput_ft(ly1,averaginglength,daterange,trendlength)

lon = np.asarray(SST2.lon)
lat = np.asarray(SST2.lat)

X1atest = preprocessing.memsplit(SST1a,testens)
X1btest = preprocessing.memsplit(SST1b,testens)
X2test = preprocessing.memsplit(SST2,testens)

X1test = np.concatenate((X1atest,X1btest),axis=1)

latvec = np.arange(-60,70,10)
lonvec = np.arange(0,360,10)

nlat = len(latvec)
nlon = len(lonvec)

#%% choose best seeds (lowest loss on validation data 2020-2050)

valfilein = "metricsfiles/" + filefront + "val_2050.npz"
val_2050=np.load(valfilein)

loss_2050=val_2050["loss"]

nseedsel = 3
seedrank = np.argsort(loss_2050,axis=2)
bestseeds = seedrank[:,:,:nseedsel]+70

#%% and calculate everything

# metrics 2020-2050
accboo_bigboi = np.empty((nlat,nlon,nseedsel,X1test.shape[0]))+np.nan
conf_bigboi = np.empty((nlat,nlon,nseedsel,X1test.shape[0]))+np.nan

IVacc_bigboi = np.empty((nlat,nlon,nseedsel,X1test.shape[0]))+np.nan
EFacc_bigboi = np.empty((nlat,nlon,nseedsel,X1test.shape[0]))+np.nan

IVconf_bigboi = np.empty((nlat,nlon,nseedsel,X1test.shape[0]))+np.nan
EFconf_bigboi = np.empty((nlat,nlon,nseedsel,X1test.shape[0]))+np.nan

predall = np.empty((nlat,nlon,nseedsel,X1test.shape[0]))+np.nan
labelall = np.empty((nlat,nlon,nseedsel,X1test.shape[0]))+np.nan
IVpredall = np.empty((nlat,nlon,nseedsel,X1test.shape[0]))+np.nan
EFpredall = np.empty((nlat,nlon,nseedsel,X1test.shape[0]))+np.nan


for ilat,lat1 in enumerate(latvec):
    lat2 = lat1+latres
    for ilon, lon1 in enumerate(lonvec):        
        lon2 = lon1+lonres
        
        print([lat1,lon1])
        
        SSTout=preprocessing.SSToutput(trendlength,lat1,lat2,lon1,lon2,daterange)
        SSTpersistence = preprocessing.SSToutput_persistence(trendlength,lat1,lat2,lon1,lon2,daterange)
        
        if ~np.isnan(np.asarray(SSTout)[0,0]):

            ucutoff = preprocessing.calc_ucutoff_q(SSTout, trainens, date1, date2, ucutoff_q)
            lcutoff = preprocessing.calc_lcutoff_q(SSTout, trainens, date1, date2, lcutoff_q)
    
            Ytest = preprocessing.memsplit(SSTout,testens)
            Ytest = preprocessing.y_onehot(Ytest,lcutoff,ucutoff)

            for iseed in range(nseedsel):
                
                seed = bestseeds[ilat,ilon,iseed]
                print(seed)
                model,x1,x2 = loadmodel(experiment_dict,X1test.shape[1],X2test.shape[1],lat1,lon1,seed)
                
                ypredtest = model.predict({"deforced":X1test,
                                "forced":X2test})
                
                pred = np.argmax(ypredtest,axis=1)
                label = np.argmax(Ytest,axis=1)
               
                # metrics from 2020-2050
                accboo_bigboi[ilat,ilon,iseed,:] = 1*(pred==label)
                conf_bigboi[ilat,ilon,iseed,:] = np.max(ypredtest,axis=1)
                
                IVnodes,EFnodes = weights_calcs.getoutputvecs(model,x1,x2,X1test,X2test)
                
                IVsoftmax = weights_calcs.softmax(IVnodes)
                EFsoftmax = weights_calcs.softmax(EFnodes)
                
                IVpred = np.argmax(IVsoftmax,axis=1)
                EFpred = np.argmax(EFsoftmax,axis=1)
                
                IVacc_bigboi[ilat,ilon,iseed,:] = 1*(IVpred==label)
                EFacc_bigboi[ilat,ilon,iseed,:] = 1*(EFpred==label)
                
                IVconf_bigboi[ilat,ilon,iseed,:] = np.max(IVsoftmax,axis=1)
                EFconf_bigboi[ilat,ilon,iseed,:] = np.max(EFsoftmax,axis=1)
                
                predall[ilat,ilon,iseed,:] = pred
                labelall[ilat,ilon,iseed,:] = label
                IVpredall[ilat,ilon,iseed,:] = IVpred
                EFpredall[ilat,ilon,iseed,:] = EFpred
                
                
#%%

filestr = "metricsfiles/" + filefront + "allpreds.npz"

np.savez(filestr,acc=accboo_bigboi,
                 conf=conf_bigboi,
                 IVacc=IVacc_bigboi,
                 IVconf=IVconf_bigboi,
                 EFacc=EFacc_bigboi,
                 EFconf=EFconf_bigboi,
                 pred=predall,
                 label=labelall,
                 IVpred=IVpredall,
                 EFpred=EFpredall,
                 )


















