#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:03:24 2023

@author: emgordy
"""

# train optimal model!

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

sys.path.append('functions/')
# custom modules, les miennes
import preprocessing
import metrics
import ANN
import experiment_settings

import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

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
ly1 = 1
ly2 = 5
averaginglength = 10

trendlength = 10

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainensnums = [0,1,2] # which pair from each set of 10
valensnums = [3]
testensnums = [4]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainens = preprocessing.enslists(trainensnums)
valens = preprocessing.enslists(valensnums)
testens = preprocessing.enslists(testensnums)

lossfunction = losses.CategoricalCrossentropy()

def scheduler(epoch,lr):
    lr_init = 0.1
    if epoch < 30:
        return lr_init
    else:
        return 0.1*lr_init

lr_callback = callbacks.LearningRateScheduler(scheduler)

def class_weight_dictionary(Ytrain):
    Ylab = np.argmax(Ytrain,axis=1)
    n = []
    for i in range(np.max(Ylab)+1):
        n.append(len(Ylab[Ylab==i]))
    max_sample = np.max(n)
    class_weights={}
    for i in range(len(n)):
        weightloop = max_sample/(n[i])
        if weightloop > 1.111111111111:
            weightloop = 0.9*weightloop
        class_weights[i] = weightloop
    return class_weights

latvec = np.arange(-60,70,10)
lonvec = np.arange(0,360,10)

seeds = np.arange(70,80) # 10 random seeds
nvalmems = len(valens)

def trainingmodels(experiment_dict,X1train,X1val,X1test,X2train,X2val,X2test,daterange):
    
    lr_init = experiment_dict["learning_rate"]
    batch_size = experiment_dict["batch_size"]
    n_epochs = experiment_dict["n_epochs"]
    patience = experiment_dict["patience"]
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
    lcutoff_q = experiment_dict["lcutoff"]
    ucutoff_q = experiment_dict["ucutoff"]
    date1 = experiment_dict["cutoff1date"]
    date2 = experiment_dict["cutoff2date"]
    es_callback = callbacks.EarlyStopping(monitor='val_loss',patience=patience,restore_best_weights=True)

    input_shape1 = X1train.shape[1]
    input_shape2 = X2train.shape[1]

    nvalmems = len(valens)
    xvec = np.arange(daterange[0],daterange[1]-trendlength+1)
    xvecvec = np.tile(xvec,nvalmems)
    boo2050 = (xvecvec>=2020) & (xvecvec<2050)       
    
    for lat1 in latvec:
        lat2 = lat1+latres  
        
        for lon1 in lonvec:
            lon2 = lon1+lonres
            print([lat1,lon1])
    
            SSTout=preprocessing.SSToutput(trendlength,lat1,lat2,lon1,lon2,daterange)
            SSTpersistence = preprocessing.SSToutput_persistence(trendlength,lat1,lat2,lon1,lon2,daterange)
            
            if ~np.isnan(np.asarray(SSTout)[0,0]):
                Ytrain = preprocessing.memsplit(SSTout,trainens)
                Yval = preprocessing.memsplit(SSTout,valens)
                Ytest = preprocessing.memsplit(SSTout,testens)
            
                ucutoff = preprocessing.calc_ucutoff_q(SSTout, trainens, date1, date2, ucutoff_q)
                lcutoff = preprocessing.calc_lcutoff_q(SSTout, trainens, date1, date2, lcutoff_q)
    
                Ytrain = preprocessing.y_onehot(Ytrain,lcutoff,ucutoff)
                Yval = preprocessing.y_onehot(Yval,lcutoff,ucutoff)
                Ytest = preprocessing.y_onehot(Ytest,lcutoff,ucutoff)
                class_weights = class_weight_dictionary(Ytrain)
                
                for iseed,seed in enumerate(seeds):
                    
                    modelstr = ANN.multimodelstr(filefront,lat1,lat2,lon1,lon2,seed) 
                    strcheck = glob.glob(modelstr)
                    if len(strcheck)==0: # so script can be stopped and restarted
                        
                        print(modelstr)
                        tf.random.set_seed(seed)
                        np.random.seed(seed)     
                        x1,input1 = ANN.build_model_api_nobias(seed,dropout_rate,ridge_param1,hiddens1,input_shape1,model1name)
                        x2,input2 = ANN.build_model_api(seed+1,dropout_rate,ridge_param2,hiddens2,input_shape2,model2name)   
                        model = ANN.fullmodel(x1,x2,input1,input2,seed)
                        
                        model.compile(loss=lossfunction, 
                                      optimizer=optimizers.SGD(learning_rate = lr_init), 
                                      metrics = ["accuracy"]
                                      )
                        history = model.fit({model1name:X1train,
                                             model2name:X2train}, 
                                            Ytrain, 
                                            batch_size = batch_size, 
                                            epochs = n_epochs, 
                                            validation_data = ({model1name:X1val,
                                                                model2name:X2val},
                                                               Yval),  
                                            verbose = 0,
                                            callbacks=[es_callback,
                                                        lr_callback 
                                                        ],
                                            class_weight=class_weights
                                            )
                        
                        model.save_weights(modelstr)
                        
                    else:
                        print(modelstr + " exists, moving right along")



#%%
experiment_name = "predictSST_terciles20202050_singlelocation_nobias_betterpreprocessing"
experiment_dict = experiment_settings.get_experiment_settings(experiment_name)

trainingdate1 = experiment_dict["trainingdate1"]
trainingdate2 = experiment_dict["trainingdate2"]

daterange = [trainingdate1,trainingdate2]

SST1a = preprocessing.SSTinput_nft(ly1,averaginglength,daterange,trendlength)
SST1b = preprocessing.SSTinput_nft(ly2,averaginglength,daterange,trendlength)
SST2 = preprocessing.SSTinput_ft(ly1,averaginglength,daterange,trendlength)

lon = np.asarray(SST2.lon)
lat = np.asarray(SST2.lat)

X1atrain = preprocessing.memsplit(SST1a,trainens)
X1btrain = preprocessing.memsplit(SST1b,trainens)
X2train = preprocessing.memsplit(SST2,trainens)

X1aval = preprocessing.memsplit(SST1a,valens)
X1bval = preprocessing.memsplit(SST1b,valens)
X2val = preprocessing.memsplit(SST2,valens)

X1atest = preprocessing.memsplit(SST1a,testens)
X1btest = preprocessing.memsplit(SST1b,testens)
X2test = preprocessing.memsplit(SST2,testens)

X1train = np.concatenate((X1atrain,X1btrain),axis=1)
X1val = np.concatenate((X1aval,X1bval),axis=1)
X1test = np.concatenate((X1atest,X1btest),axis=1)

trainingmodels(experiment_dict,X1train,X1val,X1test,X2train,X2val,X2test,daterange)




