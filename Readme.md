# Hello! Welcome to my repository for Gordon et. al 2023 (submitted to ERL fingers crossed)

This repository contains all the code to repeat the analysis in the paper but has some requirements:

### Environment
All code is written in Python 3.9. On top of the basic packages (numpy, matplotlib, scipy), this project uses 
* [tensorflow 2](https://www.tensorflow.org/install) for creating the ANNs.
* [scikit-learn](https://scikit-learn.org/stable/install.html) for K-means clustering
* [xarray](https://docs.xarray.dev/en/stable/getting-started-guide/installing.html) for preprocessing data
* [cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html) to make map data
* [cmasher](https://cmasher.readthedocs.io/user/introduction.html#how-to-install) because I like these colorbars

### Data
This project uses sea surface temperature output from the CESM2-LE which can be accessed through the [Climate Data Gateway at NCAR](https://www.earthsystemgrid.org/dataset/ucar.cgd.cesm2le.atm.proc.monthly_ave.SST.html). This data has also been regridded to 5x5 prior to the project. LMK for help with this

### A fun-loving attitude
I feel like this one is self-explanatory

## Repo Contents
I would recommend running the files in  the following order

1. calculateSSTanoms.py takes the individual ensemble member files


