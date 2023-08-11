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

from reduce_parameters_WASP127 import *
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
    
    if  nn in ind_rem:
        continue
    
    print("ORDER",O.number)
    #Start by Boucher+21 telluric correction, + remove some extreme points
    O.W_tell,O.I_tell,O.A_cl,O.V_cl = red_func.tellurics_and_borders(O,dep_min,thres_up,N_bor)
    
    #if not enough points, we discard
    if len(O.W_tell) < Npt_lim:
        print("ORDER",O.number,"(",O.W_mean,"nm) discarded (",len(O.W_cl)," pts remaining)")
        print("DISCARDED\n")
        ind_rem.append(nn)
        continue

    O.I_tell += 1

        
    # IF we have some weird orders, we rather delete them
    XX    = np.where(np.isnan(np.log(O.I_tell)))[0]
    if len(XX) > 0:
        print("ORDER",O.number,"intractable: DISCARDED\n")
        ind_rem.append(nn)
        continue
    
    #remove one pca component to calculate how many we need to remove actually, because else the white noise map has troubles
    O.n_com = 1
    O.I_fin = O.I_tell
    O.W_fin = O.W_tell
    O.I_fin, temp = red_func.apply_PCA(O,mode_norm_pca,wpca)
    
    #PCA commands
    if mode_pca =="pca":
        if auto_tune:
            O.n_com = O.tune_pca(mode_norm_pca,factor_pca,Nmap=5)+1
        else:
            O.n_com = npca[nn]  
        O.I_fin = O.I_tell
        O.I_pca, O.proj = red_func.apply_PCA(O,mode_norm_pca,wpca)
    
    #Auto encoder
    elif mode_pca == "autoencoder":
       O.I_pca = encoder.apply_encoder(O.I_fin) 
       O.proj  = np.zeros((len(ff),len(ff)))
       
    #We do nothing more
    else:
        O.I_pca = O.I_fin
        O.proj  = np.zeros((len(ff),len(ff)))
    
    #Calculate the std
    red_func.calculate_final_metrics(O,N_px,file)
        

    O.W_fin2,O.I_pca2= O.filter_pixel(O.W_fin,O.I_pca,deg_px,sig_out)

file.close()        

list_ord_fin =  np.delete(list_ord,ind_rem)
SN_fin = np.delete(SN,ind_rem,axis=0)


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
        WW.append(O.W_fin2)
        Iend.append(O.I_pca2)
        projtot.append(O.proj)
        orders_fin.append(O.number)
orders_fin = np.array(orders_fin)
savedata = (orders_fin,WW,Iend,T_obs,phase,window,berv,vstar,SN_fin,projtot)
with open(nam_fin, 'wb') as specfile:
    pickle.dump(savedata,specfile)
print("DONE")

t1          = time.time()
print("DURATION:",(t1-t0)/60.,"min")







