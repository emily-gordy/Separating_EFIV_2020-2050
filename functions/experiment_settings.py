#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:27:02 2022

@author: emgordy
"""

def get_experiment_settings(experimentname):

    experiment_dict={
        
       
        "predictSST_terciles20202050_singlelocation_nobias_betterpreprocessing": {
            "filename": "predictSST_terciles20202050_singlelocation_nobias_betterpreprocessing_",
            "hiddens1":[20,40],
            "hiddens2":[10],
            "dropout_rate":0.2,
            "ridge_param": [0.0001,0.01],            
            "learning_rate": 0.1,
            "batch_size": 64,
            "n_epochs": 1000,
            "patience": 100,
            "model1name": "deforced",
            "model2name": "forced",
            "latres": 10,
            "lonres": 10,
            "lcutoff": 0.333333,
            "ucutoff": 0.666666,
            "cutoff1date": 2020,
            "cutoff2date": 2050,
            "trainingdate1": 1960,
            "trainingdate2": 2100,
            },

        
        
        }
        
    return experiment_dict[experimentname]
    

    
