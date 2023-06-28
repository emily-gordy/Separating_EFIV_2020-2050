#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:11:30 2022

@author: emgordy
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def getweights(model,nodestr):
    node = model.get_layer(nodestr)
    nodenft = np.asarray(node.weights[0][0])[0]
    nodeft = np.asarray(node.weights[0][1])[0]
    
    return nodenft,nodeft


def softmax(vec):
    
    expvec = np.exp(vec)
    weight = np.sum(expvec,axis=1,keepdims=True)
    
    softmaxvec = expvec/weight
    
    return softmaxvec

def getoutputs(weights,nodenft,nodeft):
    
    outputnft = np.squeeze(weights[0]*nodenft)
    outputft = np.squeeze(weights[1]*nodeft)
    
    return outputnft,outputft

def getoutputs_mapwise(weight1,weight2,weight3,a,b,c):
    
    outputa = weight1*a
    outputb = weight2*b
    outputc = weight3*c
    
    outputvec = np.concatenate((outputa,outputb,outputc),axis=1)
    
    return outputvec

def getoutputvecs(model,x1,x2,X1test,X2test):
    
    lower = getweights(model,'lower')
    middle = getweights(model,'middle')
    upper = getweights(model,'upper')

    # ivpred = x1.predict(X1test)
    # efpred = x2.predict(X2test)
    
    [a,b,c] = x1.predict(X1test)
    [d,e,f] = x2.predict(X2test)

    outputnft = getoutputs_mapwise(lower[0],middle[0],upper[0],a,b,c)
    outputft = getoutputs_mapwise(lower[1],middle[1],upper[1],d,e,f)
    
    return outputnft, outputft


