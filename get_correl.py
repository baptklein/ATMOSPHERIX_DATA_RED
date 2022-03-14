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
from correlation_fcts import *




filename   = "reduced_data_HITEMP.pkl" 
name_model = "Models/Rp_HD189_GL15A_HITEMP.txt" 
name_wav   = "Models/lambdas_HD189_GL15A_HITEMP.txt"
nam_fig    = "Correl_map_HITEMP_HD189.png"
nam_res    = "correl_map_HITEMP_HD189.pkl"
Rs         = 261127.0   # Stellar radius [km]

### Velocimetric semi-amplitude
Kpmin      = 50.0 #Jupiter
Kpmax      = 250.0#Jupiter
Nkp        = 50 ### Size of the grid
Kp         = np.linspace(Kpmin,Kpmax,Nkp)

### Mid-transit planet RV [km/s]
Vmin       = -30.0
Vmax       =  30.0
Nv         = 61
Vsys       = np.linspace(Vmin,Vmax,Nv)




### READ data
print("Read data from",filename)
with open(filename,'rb') as specfile:
    orders,WW,Ir,T_obs,phase,window,berv,vstar,airmass,SN = pickle.load(specfile)
nord     = len(orders)

### Select orders for the correlation
ord_sel    = orders
V_shift    = vstar - berv


print(nord,"orders detected")
list_ord = []
for nn in range(nord):
    O        = Order(orders[nn])
    O.W_fin  = np.array(WW[nn],dtype=float)
    O.I_pca  = np.array(Ir[nn],dtype=float)
    O.SNR    = np.array(SN[nn],dtype=float)
    O.W_mean = O.W_fin.mean()
    list_ord.append(O)
print("DONE\n")


W_mod,I_mod = np.loadtxt(name_wav),np.loadtxt(name_model)
T_depth     = 1 - (I_mod/(1e5))**(2) / Rs**(2)


for kk,O in enumerate(list_ord):
    Wmin,Wmax = 0.95*O.W_fin.min(),1.05*O.W_fin.max()
    indm      = np.where((W_mod>Wmin)&(W_mod<Wmax))[0]
    W_sel     = W_mod[indm]
    O.Wm      = W_sel
    O.Im      = 1. - T_depth[indm]


### Correlation
ind_sel = []
for kk,oo in enumerate(list_ord):
    if oo.number in ord_sel: ind_sel.append(kk)
corr = compute_correlation(np.array(list_ord)[ind_sel],window,phase,Kp,Vsys,V_shift)




#### Compute statistics and plot the map
# Indicate regions to exclude when computing the NOISE level from the correlation map
Kp_lim      = [110,190]   # Exclude this Kp range we
Vsys_lim    = [-15.,15.]
snrmap_fin  = get_snrmap(np.array(orders)[ind_sel],Kp,Vsys,corr,Kp_lim,Vsys_lim)
sig_fin     = np.sum(corr[:,:,:]/snrmap_fin,axis=2)

### Get and display statistics
p_best,K_best,K_sup,K_inf,V_best,V_sup,V_inf = get_statistics(Vsys,Kp,sig_fin)


K_cut   = 120.0
V_cut   = 0.0
ind_v   = np.argmin(np.abs(Vsys-V_cut))
ind_k   = np.argmin(np.abs(Kp-K_cut))
sn_map  = sig_fin
sn_cutx = sn_map[:,ind_v]
sn_cuty = sn_map[ind_k]
cmap    = "gist_heat"


### Plot correlation + 1D cut
plot_correlation_map(Vsys,Kp,sn_map,nam_fig,V_cut,K_cut,cmap,[],sn_cuty,20)
#plot_correlation_map(Vsys,Kp,sn_map,nam_fig,K_cut,V_cut,cmap,sn_cutx,sn_cuty,20)

### Save data
savedata = (Vsys,Kp,corr,sn_map)
with open(nam_res, 'wb') as specfile:
    pickle.dump(savedata,specfile)
print("DONE")








