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

import time
import plots
import reduce_encoder as encoder

from reduce_parameters import *
import reduce_functions as red_func



#We sstart by initializing the data
T_obs,phase,window,berv,vstar,SN,list_ord= red_func.read_data_and_create_list(filename)
#Create a list of orders to remove
nord = len(list_ord)
ind_rem = []
#Do we exclude some orders ?
if len(orders_rem)>0:
    for nn in range(len(list_ord)):
        if list_ord[nn].number in orders_rem:
            ind_rem.append(nn)
V_corr      = vstar - berv                 ### Geo-to-bary correction


 ### Get transits start and end indices, either prescribed or calculated
if not set_window:
    n_ini,n_end = red_func.get_transit_dates(window)    
else:
    n_ini,n_end = n_ini_fix,n_end_fix


t0          = time.time()
NCF         = np.zeros(nord)
file        = open(nam_info,"w")


#### Main reduction
print("START DATA REDUCTION")
for nn in range(nord):
    O         = list_ord[nn]
    if  O.number in ind_rem:
        continue
    
    print("ORDER",O.number)
    #Start by Boucher+21 telluric correction, + remove some extreme points
    O.W_cl,O.I_cl,O.A_cl,O.V_cl,ind_rem = red_func.tellurics_and_borders(O,nn,ind_rem)
    
    #Do we include stellar correction a la Brogi ?
    if corr_star:
        O.I_cl  = red_func.stellar_from_file(O,nn)
        
    #Delete master out of trnasit spectrum, in stellar and telluric fram
    O.W_sub, O.I_sub = O.master_out(V_corr,n_ini,n_end)
    
    #high pass filter: suppress modal noise
    O.W_norm1,O.I_norm1 = O.normalize(O.W_sub,O.I_sub,N_med,sig_out,N_adj,N_bor)
    ### Correct for bad pixels
    O.W_norm2,O.I_norm2= O.filter_pixel(O.W_norm1,O.I_norm1,deg_px,sig_out)


    #DO we detrend airmass ? 
    if det_airmass:
        O.I_fin = red_func.airmass_correction(O)
    else:
        O.I_fin= O.I_norm2
    O.W_fin  = O.W_norm2
        
        
    #IF we have some weird orders, we rather delete them
    XX    = np.where(np.isnan(O.I_fin[0]))[0]
    if len(XX) > 0:
        print("ORDER",O.number,"intractable: DISCARDED\n")
        ind_rem.append(nn)
        continue
    
    #PCA commands
    if mode_pca =="pca":
        O.n_com = red_func.prepare_PCA(O)
        O.I_pca, O.proj = red_func.apply_PCA(O)
    
    #Auto encoder
    elif mode_pca == "autoencoder":
       O.I_pca = encoder.apply_encoder(O.I_fin) 
       O.proj  = np.zeros((len(ff),len(ff)))
       
    #We do nothing more
    else:
        O.I_pca = O.I_fin
        O.proj  = np.zeros((len(ff),len(ff)))
    
    #Calculate the std
    red_func.calculate_final_metrics(O,file)
        

    #we can plot stuffs
    if plot_red == True and O.number == numb:
        print("Plot data reduction steps")
        lab = ["Blaze-corrected spectra","Median-corrected spectra","Normalised spectra","PCA-corrected spectra"]
        plots.plot_reduction(phase,O.W_cl,O.I_cl-O.I_cl.mean(),O.W_sub,O.I_sub-1.,O.W_fin,O.I_fin,O.W_fin,O.I_pca,lab,nam_fig)        

file.close()        

list_ord_fin =  np.delete(list_ord,ind_rem)
SN_fin = np.delete(SN,ind_rem,axis=0)


if plot_red == True:
    print("PLOT METRICS")
    plots.plot_spectrum_dispersion(list_ord_fin,nam_fig)

    

### Save data for correlation
print("\nData saved in",nam_fin)
orders_fin = []
Iend  = []
WW  = []
projtot = []
for nn in range(len(list_ord)):
    if nn in ind_rem:
        continue
    else:
        O  = list_ord[nn]
        WW.append(O.W_fin)
        Iend.append(O.I_pca)
        projtot.append(O.proj)
        orders_fin.append(O.number)
orders_fin = np.array(orders_fin)
savedata = (orders_fin,WW,Iend,T_obs,phase,window,berv,vstar,SN_fin,projtot)
with open(nam_fin, 'wb') as specfile:
    pickle.dump(savedata,specfile)
print("DONE")

t1          = time.time()
print("DURATION:",(t1-t0)/60.,"min")







