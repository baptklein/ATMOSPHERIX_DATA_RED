#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:54:11 2021

@author: florian
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.signal
from scipy import stats
import pickle
import time

#Do you want to run it in parallel ?
parallel = False
#This is just for the integration over a pixel
c0 = 299792.458
pixel_window = np.linspace(-1.17,1.17,15)
weights = scipy.signal.gaussian(15,std=1.17)
weights= np.ones(15)

#Kp intervals
Kpmin = 0.0
Kpmax =300.0
Nkp = 21
Kp = np.linspace(Kpmin,Kpmax,Nkp)

#Vsys intervals
Vmin = 0
Vmax= 60
Nv = 21
Vsys = np.linspace(Vmin,Vmax,Nv)

#Number of pkl observations files and their names
num_obs = 1
pipeline_rep = "/home/florian/Bureau/Atmosphere_SPIRou/Pipeline_git/"
file_list = [pipeline_rep+"test_nonorm.pkl",]


dire_mod = "/home/florian/Bureau/Atmosphere_SPIRou/Pipeline_git/Data_Simulator/Model/Results/to-correl/reducedGL15A_HD189_onlyH2O-VMR3-T900/"


#DO we select orders or take them all ? If True, provide your order selection
# for each observation. If an order does not exist in the pkl file, it will 
# obivously not be used but will not trigger an error.
select_ord =False
list_ord1 = [31,32,33,34]

#If false, the calculation is performed over the whole dataset. If 
#True, we only select observation that have a transit window > min_window
select_phase = True
min_window = 0.2

#Interpolation factor for the speed array. If you d'ont know what that means, choose something between 1 and 10
int_speed = 8

#Number of pixels to discard at the borders. 
nbor = 10

#Do we include the projector from Gibson 2022 ?
use_proj = True
#If we just removed the mean and std of the whole map, we can use a fast verion of the projector
#Else, it will be even longer
proj_fast = True
mode_norm_pca = "per_obs" #if proj_fast is not used, we can choose
                          #how to remove mean and std in the data before PCA. Four possibilities:
                          # "none" : data untouched.
                          # "global" : suppression of mean and division by the std of the whole data set 
                          # 'per_pix': same as global but column by colum (per pixel)
                          # 'per_obs': same as global but line by line (per observation)

#Do we select only certain orders for the plot ? 
#if yes, lili is the list oforders to select
select_plot = False
lili = np.array([48,47,46,34,33,32])

#In order to calculate the std of the map,we need to exclude 
#a zone of the Kp-Vsys map around the planet. These are the limits 
#of this rectangular zone.
Kp_min_std = 80
Kp_max_std = 160
Vsys_min_std = 20
Vsys_max_std = 40

#number of levels in the contour plot
nlevels = 15

#Do we plot the correlation map ?
plot_ccf = True

#Do we save the correlation file ? If yes, put as much files as there are observations
save_ccf = False
filesave_list = [pipeline_rep+"correlated.pkl"]

#Do we add white lines at the planet position ? 
white_lines = True
Kp_planet = 120
Vsys_planet = 30


SMALL_SIZE = 28
MEDIUM_SIZE = 32
BIGGER_SIZE = 34

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


