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
parallel = True


#This is just for the integration over a pixel
c0 = 299792.458
pixel_window = np.linspace(-1.17,1.17,15)
weights = scipy.signal.gaussian(15,std=1.17)
weights= np.ones(15)

#Kp intervals
Kpmin = 0.0
Kpmax =300.0
Nkp = 121
Kp = np.linspace(Kpmin,Kpmax,Nkp)

#Vsys intervals
Vmin = -100
Vmax= 100
Nv = 201
Vsys = np.linspace(Vmin,Vmax,Nv)

#Number of pkl observations files and their names
num_obs = 3
pipeline_rep =  "/home/florian/Bureau/Atmosphere_SPIRou/Data/WASP127/DRS_06/"
file_list = [pipeline_rep+"reduced/Mar20/WASP127_Mar20_newpipeline_onlyPCA_flux_noboucher_nodetrend_autoPCA-1_imistnone.pkl",\
             pipeline_rep+"reduced/Mar21/WASP127_Mar21_newpipeline_onlyPCA_flux_noboucher_nodetrend_autoPCA-1_imistnone.pkl",\
             pipeline_rep+"reduced/May21/WASP127_May21_newpipeline_onlyPCA_flux_noboucher_nodetrend_autoPCA-1_imistnone.pkl"]

dire_mod = "/home/florian/Bureau/Atmosphere_SPIRou/Models/WASP127/Results/to-correl/reducedWASP127_onlyH2O-VMR3_T900_nonorm/"


#DO we select orders or take them all ? If True, provide your order selection
# for each observation. If an order does not exist in the pkl file, it will 
# obivously not be used but will not trigger an error.
select_ord = True
list_ord1 = np.arange(32,80)

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
mode_norm_pca = "none" #if proj_fast is not used, we can choose
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

#Do we plot the correlation map at each obs ?
plot_ccf_indiv = True
#Do we plot the global correaltion map ? 
plot_ccf_tot = True

#Do we save the correlation file ? If yes, put as much files as there are observations
save_ccf = False
filesave_list = [pipeline_rep+"correlated.pkl"]

#Do we add white lines at the planet position ? 
white_lines = True
Kp_planet = 151
Vsys_planet = -4.5


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


