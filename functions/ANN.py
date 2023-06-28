#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:24:29 2022

@author: emgordy
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
import numpy as np

# some fun functions for generating models

# combo two model baby hell yeah


def build_model_api(seed,dropout_rate,activity_reg,hiddens,input_shape,name):
    
    dense_layers = len(hiddens)
    
    inputs = tf.keras.Input(shape=(input_shape,),name=name)
    x = layers.Dense(hiddens[0], activity_regularizer=regularizers.l2(activity_reg),
                           bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                           kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                           activation='relu')(inputs)
    for i in range(dense_layers-1):
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(hiddens[i+1],
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               activation='relu')(x) ## dense layer   
    
    a = layers.Dense(1,activation='linear',
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               activity_regularizer=regularizers.l2(activity_reg))(x)
    b = layers.Dense(1,activation='linear',
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               activity_regularizer=regularizers.l2(activity_reg))(x)
    c = layers.Dense(1,activation='linear',
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               activity_regularizer=regularizers.l2(activity_reg))(x)
    
    model = tf.keras.Model(inputs=inputs,outputs=[a,b,c])
    
    return model,inputs

def build_model_api_nobias(seed,dropout_rate,activity_reg,hiddens,input_shape,name):
    
    dense_layers = len(hiddens)
    
    inputs = tf.keras.Input(shape=(input_shape,),name=name)
    x = layers.Dense(hiddens[0], activity_regularizer=regularizers.l2(activity_reg),
                           bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                           kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                           activation='relu',
                           use_bias=False)(inputs)
    for i in range(dense_layers-1):
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(hiddens[i+1],
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               activation='relu',
                               use_bias=False)(x) ## dense layer   
    
    a = layers.Dense(1,activation='linear',
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               activity_regularizer=regularizers.l2(activity_reg),
                               use_bias=False)(x)
    b = layers.Dense(1,activation='linear',
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               activity_regularizer=regularizers.l2(activity_reg),
                               use_bias=False)(x)
    c = layers.Dense(1,activation='linear',
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               activity_regularizer=regularizers.l2(activity_reg),
                               use_bias=False)(x)
    
    model = tf.keras.Model(inputs=inputs,outputs=[a,b,c])
    
    return model,inputs




def singlenodemodel(seed,singlenodename):
    
    inputlayer1 = layers.Input(1)
    inputlayer2 = layers.Input(1)
    concatlayer = layers.Concatenate()([inputlayer1,inputlayer2])
    
    dense = layers.Dense(1,activation='linear',
                               bias_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               kernel_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                               use_bias=False,
                               # kernel_constraint=tf.keras.constraints.NonNeg(),
                               )(concatlayer)
    model = tf.keras.Model(inputs=[inputlayer1,inputlayer2],outputs=dense,name=singlenodename)
    
    return model


def finallayer():
    
    inputlayer1 = layers.Input(1,)
    inputlayer2 = layers.Input(1,)
    inputlayer3 = layers.Input(1,)

    concatlayer = layers.Concatenate()([inputlayer1,inputlayer2, inputlayer3])
    final = tf.keras.layers.Softmax()(concatlayer)
    lastmodel = tf.keras.Model(inputs=[inputlayer1, inputlayer2, inputlayer3], outputs = final)
    
    return lastmodel


def fullmodel(x1,x2,input1,input2,seed):

    a,b,c = x1(input1) # define the outputs of individual NNs
    d,e,f = x2(input2)
    
    singlenodemodel1 = singlenodemodel(seed,'lower') #define a model that takes two inputs to 
    singlenodemodel2 = singlenodemodel(seed,'middle') #a single node with linear activation
    singlenodemodel3 = singlenodemodel(seed,'upper')
    
    singlenodea = singlenodemodel1([a, d]) #pass the individual NN outputs in pairs to
    singlenodeb = singlenodemodel2([b, e]) #to the corresponding node
    singlenodec = singlenodemodel3([c, f])
    
    lastmodel = finallayer() # define a model that takes nodes, concats and adds a softmax    
    finaloutput = lastmodel([singlenodea, singlenodeb, singlenodec]) # point single nodes to the model
    
    model = tf.keras.Model(inputs = [input1, input2], # and finally point the initial first layer 
                           outputs = [finaloutput]) # to the final final layer!
    
    return model

def multimodelstr(filefront,lat1,lat2,lon1,lon2,seed):
    
    strout = "models/" + filefront + "lat=%d-%d_lon%d-%d_seed=%d.h5" %(lat1,lat2,lon1,lon2,seed)
    
    return strout


def multimodelstr_sweep(filefront,lat1,lat2,lon1,lon2,seed,sweepvar,sweepval):
    
    strout = "models/" + filefront + "lat=%d-%d_lon%d-%d_seed=%d_%f=" %(lat1,lat2,lon1,lon2,seed,sweepval) + sweepvar + ".h5" 
    
    return strout

def singlemodel_concatout(x,inputs):
    
    node1,node2,node3 = x(inputs)    
    concatlayer = layers.Concatenate()([node1,node2,node3])
    
    xconcat = tf.keras.Model(inputs=inputs,outputs=concatlayer)
    
    return xconcat

# code from Tony's baselines work
def get_gradients(model, inputs, top_pred_idx=None):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        model (tensorflow model): to pull gradients from
        inputs: 2D/3D/4D matrix of samples
        top_pred_idx: (optional) Predicted label for the x_data
                      if classification problem. If regression,
                      do not include.

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    inputs = tf.cast(inputs, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        
        # Run the forward pass of the layer and record operations
        # on GradientTape.
        preds = model(inputs, training=False)  
        
        # For classification, grab the top class
        if top_pred_idx is not None:
            preds = preds[:, top_pred_idx]
        
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.        
    grads = tape.gradient(preds, inputs)
    return grads

def get_integrated_gradients(inputs, model, baseline=None, num_steps=50, top_pred_idx=None):
    """Computes Integrated Gradients for a prediction.

    Args:
        inputs (ndarray): 2D/3D/4D matrix of samples
        model (tensorflow model): the NN to pull gradients from
        baseline (ndarray): The baseline image to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        top_pred_idx: (optional) Predicted label for the x_data
                      if classification problem. If regression,
                      do not include.            

    Returns:
        Integrated gradients w.r.t input image
    """
    # If baseline is not provided, start with zeros
    # having same size as the input image.
    if baseline is None:
        input_size = np.shape(inputs)[1:]
        baseline = np.zeros(input_size).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # 1. Do interpolation.
    inputs = inputs.astype(np.float32)
    interpolated_inputs = [
        baseline + (step / num_steps) * (inputs - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_inputs = np.array(interpolated_inputs).astype(np.float32)

    # 3. Get the gradients
    grads = []
    for i, x_data in enumerate(interpolated_inputs):
        grad = get_gradients(model, x_data, top_pred_idx=top_pred_idx)     
        grads.append(grad)
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    # 4. Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)
    # 5. Calculate integrated gradients and return
    integrated_grads = (inputs - baseline) * avg_grads
    return integrated_grads


def get_input_t_gradient(inputs, model, top_pred_idx=None):
    """Computes Input times gradient for a prediction.

    Args:
        inputs (ndarray): 2D/3D/4D matrix of samples
        model (tensorflow model): the NN to pull gradients from
        top_pred_idx: (optional) Predicted label for the x_data
                      if classification problem. If regression,
                      do not include.            

    Returns:
        Input times gradient w.r.t input image
    """

    inputs = inputs.astype(np.float32)

    grads = get_gradients(model, inputs, top_pred_idx=top_pred_idx)
    
    input_t_gradient = np.multiply(inputs,grads)
    
    return input_t_gradient

















