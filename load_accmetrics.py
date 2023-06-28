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

ntestmems = len(testens)
xvec = np.arange(daterange[0],daterange[1]-trendlength+1)
xvecvec = np.tile(xvec,ntestmems)
boo2050 = (xvecvec>=2020) & (xvecvec<2050)
boo19602020 = (xvecvec>=1960) & (xvecvec<=2010)

#%% choose best seeds (lowest loss on validation data 2020-2050)

valfilein = "metricsfiles/" + filefront + "val_2050.npz"
val_2050=np.load(valfilein)

loss_2050=val_2050["loss"]

nseedsel = 3
seedrank = np.argsort(loss_2050,axis=2)
bestseeds = seedrank[:,:,:nseedsel]+70

seedfile = "metricsfiles/" + filefront + "bestseeds.npz"
np.savez(seedfile,seedrank=seedrank,)

nconfs = 10
nshuffle = 500

#%% and calculate everything

# metrics 2020-2050
acctest_2050 = np.empty((nlat,nlon,nseedsel,nconfs))+np.nan
losstest_2050 = np.empty((nlat,nlon,nseedsel))+np.nan
efonlytest_2050 = np.empty((nlat,nlon,nseedsel,nconfs))+np.nan
ivonlytest_2050 = np.empty((nlat,nlon,nseedsel,nconfs))+np.nan
freqclass_2050 = np.empty((nlat,nlon,nseedsel,4))+np.nan
accclass_2050 = np.empty((nlat,nlon,nseedsel,4))+np.nan
shuffleacc_2050 = np.empty((nlat,nlon,nseedsel,10))+np.nan
shuffle90_2050 = np.empty((nlat,nlon,nseedsel,10))+np.nan
persistencetest_2050 = np.empty((nlat,nlon))+np.nan
zeroacc_2050 = np.empty((nlat,nlon,nseedsel,nconfs))+np.nan

# metrics across entire timeseries
acctest = np.empty((nlat,nlon,nseedsel,nconfs))+np.nan
losstest = np.empty((nlat,nlon,nseedsel))+np.nan
efonlytest = np.empty((nlat,nlon,nseedsel,nconfs))+np.nan
ivonlytest = np.empty((nlat,nlon,nseedsel,nconfs))+np.nan
freqclass = np.empty((nlat,nlon,nseedsel,4))+np.nan
accclass = np.empty((nlat,nlon,nseedsel,4))+np.nan
shuffleacc = np.empty((nlat,nlon,nseedsel,10))+np.nan
shuffle90 = np.empty((nlat,nlon,nseedsel,10))+np.nan
persistencetest = np.empty((nlat,nlon))+np.nan
zeroacc = np.empty((nlat,nlon,nseedsel,nconfs))+np.nan
randomchance = np.empty((nlat,nlon))+np.nan
ucutoffval = np.empty((nlat,nlon))+np.nan
lcutoffval = np.empty((nlat,nlon))+np.nan

# metrics across observational record
acctest_19602020 = np.empty((nlat,nlon,nseedsel,nconfs))+np.nan
losstest_19602020 = np.empty((nlat,nlon,nseedsel))+np.nan
persistencetest_19602020 = np.empty((nlat,nlon))+np.nan
randomchance_19602020 = np.empty((nlat,nlon))+np.nan

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
            
            Ytestpersistence = preprocessing.memsplit(SSTpersistence,testens)
            Ytestpersistence = preprocessing.y_onehot(Ytestpersistence,lcutoff,ucutoff)
            
            randomchance[ilat,ilon] = metrics.randomchanceacc(Ytest)
            randomchance_19602020[ilat,ilon] = metrics.randomchanceacc(Ytest[boo19602020])
            ucutoffval[ilat,ilon] = ucutoff
            lcutoffval[ilat,ilon] = lcutoff
            
            persistencetest[ilat,ilon] = metrics.persistence(Ytestpersistence,Ytest)
            persistencetest_2050[ilat,ilon] = metrics.persistence(Ytestpersistence[boo2050],Ytest[boo2050])
            persistencetest_19602020[ilat,ilon] = metrics.persistence(Ytestpersistence[boo19602020],Ytest[boo19602020])
            
            for iseed in range(nseedsel):
                
                seed = bestseeds[ilat,ilon,iseed]
                print(seed)
                model,x1,x2 = loadmodel(experiment_dict,X1test.shape[1],X2test.shape[1],lat1,lon1,seed)
                
                ypredtest = model.predict({"deforced":X1test,
                                "forced":X2test})
               
                # metrics from 2020-2050
                acctest_2050[ilat,ilon,iseed,:] = metrics.confxacc(ypredtest[boo2050],Ytest[boo2050])
                losstest_2050[ilat,ilon,iseed] = metrics.loss(ypredtest[boo2050],Ytest[boo2050])  
                
                IVnodes,EFnodes = weights_calcs.getoutputvecs(model,x1,x2,X1test[boo2050],X2test[boo2050])
                efonlytest_2050[ilat,ilon,iseed,:] = metrics.singlemodelaccxconf(EFnodes,Ytest[boo2050])
                ivonlytest_2050[ilat,ilon,iseed,:] = metrics.singlemodelaccxconf(IVnodes,Ytest[boo2050])                  
                
                freqclass_2050[ilat,ilon,iseed,:] = metrics.predclass(IVnodes,EFnodes,ypredtest[boo2050])
                accclass_2050[ilat,ilon,iseed,:] = metrics.accclass(IVnodes,EFnodes,ypredtest[boo2050],Ytest[boo2050])

                # metrics across full time series
                acctest[ilat,ilon,iseed,:] = metrics.confxacc(ypredtest,Ytest)
                losstest[ilat,ilon,iseed] = metrics.loss(ypredtest,Ytest)  
                
                IVnodes,EFnodes = weights_calcs.getoutputvecs(model,x1,x2,X1test,X2test)
                efonlytest[ilat,ilon,iseed,:] = metrics.singlemodelaccxconf(EFnodes,Ytest)
                ivonlytest[ilat,ilon,iseed,:] = metrics.singlemodelaccxconf(IVnodes,Ytest)  
                
                freqclass[ilat,ilon,iseed,:] = metrics.predclass(IVnodes,EFnodes,ypredtest)
                accclass[ilat,ilon,iseed,:] = metrics.accclass(IVnodes,EFnodes,ypredtest,Ytest)
                
                # shuffling (done simultaneously to speed up computation)
                shuffledist,shuffledist_2050 = metrics.shuffleaccdist_drawdist_wboo(nshuffle,X1test,X2test,Ytest,model,seed,boo2050)
                
                shuffleacc_2050[ilat,ilon,iseed,:] = np.mean(shuffledist_2050,axis=0)
                shuffle90_2050[ilat,ilon,iseed,:] = np.percentile(shuffledist_2050,90,axis=0)  
                
                shuffleacc[ilat,ilon,iseed,:] = np.mean(shuffledist,axis=0)
                shuffle90[ilat,ilon,iseed,:] = np.percentile(shuffledist,90,axis=0)              
                 
                acctest_19602020[ilat,ilon,iseed,:] = metrics.confxacc(ypredtest[boo19602020,:],Ytest[boo19602020,:])
                losstest_19602020[ilat,ilon,iseed] = metrics.loss(ypredtest[boo19602020,:],Ytest[boo19602020,:])
                
                # metrics for inputting zeros only
                ypredzeros = model.predict({"deforced":np.zeros(X1test.shape),
                                            "forced":X2test})
                
                zeroacc[ilat,ilon,iseed,:] = metrics.confxacc(ypredzeros,Ytest)
                zeroacc_2050[ilat,ilon,iseed,:] = metrics.confxacc(ypredzeros[boo2050],Ytest[boo2050])
                
#%% save metrics

file_2050 = "metricsfiles/" + filefront + "test_2050.npz"
file_all = "metricsfiles/" + filefront + "test_full.npz"
file_19602020 = "metricsfiles/" + filefront + "test_19602020.npz"

np.savez(file_2050,acc=acctest_2050,
                loss=losstest_2050,
                efonly=efonlytest_2050,
                ivonly=ivonlytest_2050,
                freqclass=freqclass_2050,
                accclass=accclass_2050,
                shuffleacc=shuffleacc_2050,
                shuffle90=shuffle90_2050,
                acczeros=zeroacc_2050,
                persistence=persistencetest_2050
                )

np.savez(file_all,acc=acctest,
                loss=losstest,
                efonly=efonlytest,
                ivonly=ivonlytest,
                freqclass=freqclass,
                accclass=accclass,
                shuffleacc=shuffleacc,
                shuffle90=shuffle90,
                acczeros=zeroacc,
                lcutoff=lcutoffval,
                ucutoff=ucutoffval,
                randomchance=randomchance,
                persistence=persistencetest,
                )


np.savez(file_19602020,acc=acctest_19602020,
                loss=losstest_19602020,
                randomchance=randomchance_19602020,
                persistence=persistencetest_19602020,
                )





           
