#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/2022 12:54:11 2021

@author: Baptiste & Florian
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import time
from functions import *


### Name of the picle file to read the data from
filename = "Simu_HD189_HITEMP.pkl" 
nam_fin  = "reduced_data_HITEMP.pkl"


### Read data in pickle format
### Namely: 
#           - orders: List of orders -- absolute nbs
#           - WW:     Wavelength vector for each order [list obj (N_order,N_wav)]
#           - Ir:     Intensity values for each order [list of 2D arrays (N_order,N_obs,N_wav)]
#           - blaze:  Blaze values for each order [list of 2D arrays (N_order,N_obs,N_wav)]
#           - Ia:     Telluric spectra for each order [list of 2D arrays (N_order,N_obs,N_wav)]
#           - T_obs:  Observation dates [BJD]
#           - phase:  Planet orbital phase - centered on mid-transit
#           - window: Transit window (1 --> mid-transit; 0 --> out-of-transit)
#           - berv:   BERV values [km/s]
#           - vstar:  Stellar RV [km/s]
#           - airmass:Airmass values
#           - SN:     Signal-to-noise values for each order [N_order,N_obs]
print("Read data from",filename)
with open(filename,'rb') as specfile:
    orders,WW,Ir,blaze,Ia,T_obs,phase,window,berv,vstar,airmass,SN = pickle.load(specfile)


### Data reduction parameters
dep_min  = 0.3       # remove all data when telluric relative absorption > 1 - dep_min
thres_up = 0.05      # Remove the line until reaching 1-thres_up
Npt_lim  = 1000       # If the order contains less than Npt_lim points, it is discarded from the analysis

### Interpolation parameters
pixel    = np.linspace(-1.13,1.13,11)   ### Sampling a SPIRou pixel in velocity space -- Width ~ 2.28 km/s
kind     = "linear"                     ### scipy interp1d parameter
N_bor    = 15                           ### Nb of pts removed at each extremity (twice)

### Normalisation parameters
N_med    = 150                          ### Nb of points used in the median filter for the inteprolation
sig_out  = 5.0                          ### Threshold for outliers identification during normalisation process 
deg_px   = 2                            ### Degree of the polynomial fit to the distribution of pixel STDs

### Parameters for detrending with airmass
det_airmass = True
deg_airmass = 2

### Parameters PCA
mode_pca    = "pca"                     ### "pca"/"PCA" or "autoencoder"
npca        = np.array(1*np.ones(len(orders)),dtype=int)      ### Nb of removed components
                                        ### Note: Automatic mode to be implemented


    
    
    
### Create order objects
nord     = len(orders)
print(nord,"orders detected")
list_ord = []
for nn in range(nord):
    O        = Order(orders[nn])
    O.W_raw  = np.array(WW[nn],dtype=float)
    O.I_raw  = np.array(Ir[nn],dtype=float)
    O.blaze  = np.array(blaze[nn],dtype=float)    
    O.I_atm  = np.array(Ia[nn],dtype=float)
    O.SNR    = np.array(SN[nn],dtype=float)
    O.W_mean = O.W_raw.mean()
    list_ord.append(O)
print("DONE\n")




ind_rem     = []
V_corr      = vstar - berv                  ### Geo-to-bary correction
n_ini,n_end = get_transit_dates(window) ### Get transits start and end indices
c0          = Constants().c0
t0          = time.time()

#### Main reduction
print("START DATA REDUCTION")
for nn in range(nord):

    O         = list_ord[nn]
    print("ORDER",O.number)

    ### First we identify strong telluric lines and remove the data within these lines -- see Boucher+2021
    W_cl,I_cl =  O.remove_tellurics(dep_min,thres_up)
    
    ### If the order does not contain enough points, it is discarded
    if len(W_cl) < Npt_lim:
        print("ORDER",O.number,"(",O.W_mean,"nm) discarded (",len(W_cl)," pts remaining)")
        print("DISCARDED\n")
        ind_rem.append(nn)
    else:
        print(len(O.W_raw)-len(W_cl),"pts removed from order",O.number,"(",O.W_mean,"nm) -- OK")
        
        
        ### If the order is kept - Remove high-SNR out-of-transit reference spectrum    
        ### Start by computing mean spectrum in the stellar rest frame
        V_cl      = c0*(W_cl/O.W_mean-1.)
        I_bary    = move_spec(V_cl,I_cl,V_corr,pixel,kind)  ## Shift to stellar rest frame      
        I_med     = np.median(np.concatenate((I_bary[:n_ini],I_bary[n_end:]),axis=0),axis=0) ## Compute median out-of-transit        
        I_med_geo = move_spec(V_cl,np.array([I_med]),-1.*V_corr,pixel,kind)  ## Move back ref spectrum to Geocentric frame
        I_sub1    = np.zeros(I_cl.shape)
        for kk in range(len(I_cl)):
            X          = np.array([np.ones(len(I_med_geo[kk])),I_med_geo[kk]],dtype=float).T
            p,pe       = LS(X,I_cl[kk])
            Ip         = np.dot(X,p)
            I_sub1[kk] = I_cl[kk]/Ip
            
        ### Then compute reference spectrum in the Geocentric frame
        I_med2  = np.median(np.concatenate((I_sub1[:n_ini],I_sub1[n_end:]),axis=0),axis=0) 
        I_sub2  = np.zeros(I_sub1.shape)
        for kk in range(len(I_sub1)):
            X          = np.array([np.ones(len(I_med2)),I_med2],dtype=float).T
            p,pe       = LS(X,I_sub1[kk])
            Ip         = np.dot(X,p)
            I_sub2[kk] = I_sub1[kk]/Ip    
            
        ### Remove extremities to avoid interpolation errors
        W_sub = W_cl[N_bor:-N_bor]
        I_sub = I_sub2[:,N_bor:-N_bor]
        ### END of STEP 1    
        
        
        ### STEP 2 -- NORMALISATION AND OUTLIER REMOVAL
        W_norm1,I_norm1 = O.normalize(W_sub,I_sub,N_med,sig_out,1,N_bor)
        ### Correct for bad pixels
        W_norm2,I_norm2 = O.filter_pixel(W_norm1,I_norm1,deg_px,sig_out)
        ### END of STEP 2    
        
        
        ### STEP 3 -- DETREND WITH AIRMASS -- OPTIONAL
        if det_airmass:
            I_log           = np.log(I_norm1)
            I_det_log       = O.detrend_airmass(W_norm2,I_norm2,airmass,deg_airmass)
            I_det           = np.exp(I_det_log)
            O.I_fin         = I_det
        else:
            O.I_fin         = I_norm2
        O.W_fin  = W_norm2  
        O.W_bary = []
        for uu in range(len(O.I_fin)):
            O.W_bary.append(O.W_fin/(1.0 + V_corr[uu]/c0))
        O.W_bary = np.array(O.W_bary,dtype=float)
        
                      
        
        
        ### STEP 4 -- REMOVING CORRELATED NOISE -- PCA/AUTOENCODERS
        Il    = np.log(O.I_fin)
        im    = np.nanmean(Il)
        ist   = np.nanstd(Il)        
        ff    = (Il - im)/ist
        
        XX    = np.where(np.isnan(O.I_fin[0]))[0]
        if len(XX) > 0:
            print("ORDER",O.number,"intractable: DISCARDED\n")
            ind_rem.append(nn)
        else:
            if mode_pca == "pca" or mode_pca == "PCA":
                pca   = PCA(n_components=npca[nn])
                x_pca = np.float32(ff)
                pca.fit(x_pca)
                principalComponents = pca.transform(x_pca)
                x_pca_projected = pca.inverse_transform(principalComponents)        
                O.I_pca = (ff-x_pca_projected)*ist+im
            
                
            ### ESTIMATES FINAL METRICS
            N_px          = 200
            indw          = np.argmin(np.abs(O.W_fin-O.W_fin.mean())) 
            O.SNR_mes     = 1./np.std(O.I_fin[:,indw-N_px:indw+N_px],axis=1) 
            O.SNR_mes_pca = 1./np.std(O.I_pca[:,indw-N_px:indw+N_px],axis=1)        
            

        
print("DATA REDUCTION DONE\n")        
        

### Plot final metrics -- RMS per spectrum in each order
print("PLOT METRICS")
orders_fin   = np.delete(orders,ind_rem)
list_ord_fin = np.delete(list_ord,ind_rem)
nam_fig      = "spectrum_dispersion.png"
plot_spectrum_dispersion(list_ord_fin,nam_fig)
print("DONE\n")

### Save data for correlation
print("\nData saved in",nam_fin)
Ir  = []
WW  = []
for nn in range(len(orders_fin)):
    O  = list_ord_fin[nn]
    WW.append(O.W_fin)
    Ir.append(O.I_pca)
savedata = (orders_fin,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN)
with open(nam_fin, 'wb') as specfile:
    pickle.dump(savedata,specfile)
print("DONE")

t1          = time.time()
print("DURATION:",(t1-t0)/60.,"min")







