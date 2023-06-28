#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 13:20:45 2022

@author: emgordy
"""

import xarray as xr
import numpy as np
import glob
import tensorflow as tf
from scipy.stats import binom
import gc

def confxacc(ypred,ytrue):
    labels = np.argmax(ytrue,axis=1)
    preds = np.argmax(ypred,axis=1)
    predval = np.max(ypred,axis=1)    
    
    percentiles = np.arange(0,100,10)
    
    accpercentiles = np.empty(10)
    
    for iper,per in enumerate(percentiles):
        thres = np.percentile(predval,per)
        thresboo = predval>=thres
        predloop = preds[thresboo]
        labelloop = labels[thresboo]
        ncorr = labelloop[predloop==labelloop].shape[0]
        nloop = labelloop.shape[0]
        
        accpercentiles[iper] = ncorr/nloop
        
    return accpercentiles

def loss(ypred,ytrue):
    
    lossout = np.mean(-np.sum(ytrue*np.log(ypred),axis=1))
    
    return lossout
        
def randomchanceacc(ytrue):
    
    # random chance for unbalanced classes
    maxclass = np.max(np.sum(ytrue,axis=0))
    accout = maxclass/ytrue.shape[0]
    
    return accout


def shuffleaccxconf(nshuffle,X1,X2,ytrue,model):
    
    loopacc = np.empty((nshuffle,10))
    nX1 = X2.shape[0] # grab from full X1 distribution, but if X2 is subset then grab only that number
    inds = np.arange(nX1)
    for ishuffle in range(nshuffle):
        randinds=np.random.choice(inds,size=nX1,replace=False)
        X1in = X1[randinds]
        ypred = model.predict({'deforced':X1in,
                        'forced':X2})
        loopacc[ishuffle,:] = confxacc(ypred,ytrue)
    
    shuffleacc = np.mean(loopacc,axis=0)
    shuffle5 = np.percentile(loopacc,5,axis=0)
    shuffle95 = np.percentile(loopacc,95,axis=0)
    
    _ = gc.collect()
    
    return shuffleacc,shuffle5,shuffle95

def persistence(ypersistence,ytrue):
    
    labels = np.argmax(ytrue,axis=1)
    persistencepred = np.argmax(ypersistence,axis=1)
    
    percorr = labels==persistencepred
    
    peracc = len(percorr[percorr])/len(percorr)
    
    return peracc    

def predclass(IVnodes,EFnodes,ypred):

    IVlargest = np.argmax(IVnodes,axis=1)
    EFlargest = np.argmax(EFnodes,axis=1)

    pred = np.argmax(ypred,axis=1)

    IVhighboo = IVlargest==pred
    EFhighboo = EFlargest==pred

#     IVwin = (IVnodes[rowind,pred])>(EFnodes[rowind,pred])

    classes = np.empty((len(pred)))+np.nan

    classes[IVhighboo & EFhighboo] = 1
    classes[IVhighboo & ~EFhighboo] = 2
    classes[~IVhighboo & EFhighboo] = 3
    classes[~IVhighboo & ~EFhighboo] = 4
    
    classfrac=np.empty(4)
    
    for i in range(4):
        classfrac[i] = len(classes[classes==(i+1)])/len(classes)
    
    return classfrac

def accclass(IVnodes,EFnodes,ypred,ytrue):

    IVlargest = np.argmax(IVnodes,axis=1)
    EFlargest = np.argmax(EFnodes,axis=1)

    pred = np.argmax(ypred,axis=1)
    label = np.argmax(ytrue,axis=1)
    modelcorr = pred==label

    IVhighboo = IVlargest==pred
    EFhighboo = EFlargest==pred

    classes = np.empty((len(pred)))+np.nan

    classes[IVhighboo & EFhighboo] = 1
    classes[IVhighboo & ~EFhighboo] = 2
    classes[~IVhighboo & EFhighboo] = 3
    classes[~IVhighboo & ~EFhighboo] = 4
    
    classacc=np.empty(4)
    for i in range(4):
        if len(classes[classes==(i+1)]) != 0:
            classacc[i] = len(classes[(classes==(i+1)) & modelcorr])/len(classes[classes==(i+1)])
        else:
            classacc[i]=np.nan
        
    return classacc
    
def shuffleaccdist(nshuffle,X1,X2,ytrue,model):
    
    loopacc = np.empty((nshuffle,10))
    nX1 = X2.shape[0]
    inds = np.arange(nX1)
    for ishuffle in range(nshuffle):
        randinds=np.random.choice(inds,size=nX1,replace=False)
        X1in = X1[randinds]
        ypred = model.predict({'deforced':X1in,
                        'forced':X2})
        loopacc[ishuffle,:] = confxacc(ypred,ytrue)
        _ = gc.collect()
    
    return loopacc 



# def shuffleaccdist_1d(nshuffle,X1,X2,ytrue,model,seed):
    
#     loopacc = np.empty((nshuffle,10))
#     # nX1 = X2.shape[0]
#     # inds = np.arange(nX1)
#     rng = np.random.default_rng()
#     for ishuffle in range(nshuffle):
#         # randinds=np.random.choice(inds,size=nX1,replace=False)
#         # X1in = X1[randinds]
        
#         X1in = rng.permuted(X1,axis=1)
        
#         ypred = model.predict({'deforced':X1in,
#                         'forced':X2})
#         loopacc[ishuffle,:] = confxacc(ypred,ytrue)
    
    
#     _ = gc.collect()
    
#     return loopacc 

def singlemodelaccxconf(modelnodes,ytrue):
    
    ypred = np.exp(modelnodes)/np.sum(np.exp(modelnodes),axis=1,keepdims=True)
    modelonlyacc = confxacc(ypred,ytrue)
    
    return modelonlyacc

# def shuffleaccdist_drawdist(nshuffle,X1,X2,ytrue,model,seed):
    
#     loopacc = np.empty((nshuffle,10))
    
#     for ishuffle in range(nshuffle):
        
#         inds = np.random.rand(*X1.shape).argsort(0)
#         X1loop = X1[inds, np.arange(X1.shape[1])]
        
#         ypred = model.predict({'deforced':X1loop,
#                                'forced':X2},
#                                 verbose=0)
#         loopacc[ishuffle,:] = confxacc(ypred,ytrue)    
#         _ = gc.collect()
    
#     return loopacc 

def shuffleaccdist_drawdist_wboo(nshuffle,X1,X2,ytrue,model,seed,boo):
    
    loopacc = np.empty((nshuffle,10))
    loopacc_boo = np.empty((nshuffle,10))
    
    for ishuffle in range(nshuffle):
        
        inds = np.random.rand(*X1.shape).argsort(0)
        X1loop = X1[inds, np.arange(X1.shape[1])]
        
        ypred = model.predict({'deforced':X1loop,
                               'forced':X2},
                                verbose=0)
        loopacc[ishuffle,:] = confxacc(ypred,ytrue)    
        loopacc_boo[ishuffle,:] = confxacc(ypred[boo],ytrue[boo])    

        _ = gc.collect()
    
    return loopacc,loopacc_boo 

























