#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:44:37 2021

@author: florian
"""
import numpy as np
from numpy import ma

import sys
import os
import time

from scipy import signal
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.optimize import minimize

from sklearn.decomposition import PCA

import astropy
from astropy.io import fits
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting, polynomial

import matplotlib.pyplot as plt
import matplotlib
import pickle

def wavelength_to_speed(W,Vc,berv,nobs):
    """
    Shift spectra in the stellar rest frame
    Convert wavelengths into velocities (using the self.W_raw to be consistent with the model)
    The apply rv_s - rv_ber and store all into self.V_corr (corrected velocity) and self.W_corr (corrected wavelengths)
    """

                          #Speed of light
    Wmean = np.mean(W)
    ### Correct for stellar and barycentric Earth radial velocities
    V_corr      = c0*(W/Wmean - 1.)

    return V_corr  ### warning: 2D matrice

c0          = 299792.458 

REP_DATA = "/home/florian/Bureau/Atmosphere_SPIRou/Data/Gl15A/reduced/pkl/double/"
filename = "Simu_HD189_EXOMOL_2.pkl"


dir_res = "/home/florian/Bureau/Atmosphere_SPIRou/Data/Gl15A/to_mcmc/pkl/double/PCA1/"
name_fin = "test_GL15A_2.pkl"

with open(REP_DATA+filename,'rb') as ccfile:
    orders,W_data,I_data,T_obs,phase,window,berv,Vc,airmass,SN = pickle.load(ccfile)


lambdas = np.zeros((2,130))
orders = np.array([79,78])

V_data = []
Std_tot = []
Vstar = Vc-berv
Wmean_tot = []


for k in range(len(orders)):
    Wmean = np.mean(W_data[k])
    V_corr =c0*(W_data[k]/Wmean - 1.)
    V_data.append(V_corr)
    lambdas[:,orders[k]] = [W_data[k][0],W_data[k][-1]]
    Wmean_tot.append(Wmean)

    #We fit a second order parabola to the Std data
    Std = np.zeros(np.shape(I_data[k])[1])
    for i in range(np.shape(I_data[k])[1]):
        Std[i] = (np.std(I_data[k][:,i]))
    fit2 = np.polyfit(V_corr,Std,2)
    Stdfit = np.poly1d(fit2)
    Std_tot.append(Std)
#    Std_tot.append(Stdfit(V_corr))
    
    
savedata = (orders,Wmean_tot,V_data,I_data,Std_tot,phase,window,Vstar,)
with open(dir_res+name_fin, 'wb') as specfile:
    pickle.dump(savedata,specfile)    
    











