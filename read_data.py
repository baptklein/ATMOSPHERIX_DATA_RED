"""
Created on 03/2022 12:54:11 2021

This script performs data processing and analysis for a transit observation.
Authors: Baptiste & Florian
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import read_functions as read_func
from read_parameters import *

### READ FITS FILES
print("\nRead data from",dir_data)
list_ord = []
nord     = len(orders)
for kk in range(nord):
    list_ord.append(read_func.Order(orders[kk])) ### Initialize list of Order objects
list_ord,airmass,T_obs,berv,snr_mat = read_func.read_data_spirou(dir_data,list_ord,nord)
nobs = len(T_obs)
print("DONE")


### Pre-process data: correct for blaze and remove NaNs
print("\nRemove NaNs")
cmp = 0
for mm in range(nord):
    O   = list_ord[cmp]   
    err = O.remove_nan()
    if err > 0: ### If only NaNs
        print("Order",O.number,"empty - removed")
        del orders[cmp]
        del list_ord[cmp]
    else: cmp += 1
nord = len(list_ord)
print("DONE")


print("\nCompute transit")
### Compute phase
phase  = (T_obs - T0)/Porb
phase -= int(phase[-1])  


### Compute transit window
flux     = read_func.compute_transit(Rp,Rs,ip,T0,ap,Porb,ep,wp,ld_mod,ld_coef,T_obs)
window       = (1-flux)/np.max(1-flux)
print("DONE")

### Compute Planet-induced RV
Vp           = read_func.get_rvs(T_obs,Ks,Porb,T0)
Vc           = V0 + Vp - berv  #Geocentric-to-barycentric correction



### Plot transit information
if plot:
    print("\nPlot transit")
    TT     = 24.*(T_obs - T0)
    ypad   = 15  # pad of the y label
    plt.figure(figsize=(15,12))
    # Transit flux
    ax  = plt.subplot(411)
    ax.plot(TT,flux,"-+r",label="planet")
    plt.legend(loc=3,fontsize=16)
    ax.set_ylabel("Transit curve\n", labelpad=ypad)
    # Airmass
    ax = plt.subplot(412)
    plt.plot(TT,airmass,"-k")
    ax.set_ylabel("Airmass\n", labelpad=ypad)
    # RV correction between Geocentric frame and stellar rest frame
    ax = plt.subplot(413)
    plt.plot(TT,Vc,"-k")
    ax.set_ylabel("RV correction\n[km/s]", labelpad=ypad)
    # Maximum S/N
    ax = plt.subplot(414)
    plt.plot(TT,np.max(snr_mat,axis=1),"+k")
    plt.axhline(np.mean(np.max(snr_mat,axis=1)),ls="--",color="gray")
    plt.xlabel("Time wrt transit [h]")
    ax.set_ylabel("Peak S/N\n", labelpad=ypad)
    plt.subplots_adjust(hspace=0.02)
    plt.savefig(dir_figures+"transit_info.pdf",bbox_inches="tight")
    plt.close()
    print("DONE")


### Save as pickle
print("\nData saved in",name_fin)
Ir  = []
Ia  = []
Bl  = []
Ip1 = []
Ip2 = []
WW  = []
SN  = []
for nn in range(nord):
    O  = list_ord[nn]
    WW.append(O.W_raw)
    Ir.append(O.I_raw)
    Ia.append(O.I_atm)
    Bl.append(O.B_raw)
    SN.append(O.SNR)
    
### Namely: 
#           - orders: List of orders -- absolute nbs
#           - WW:     Wavelength vector for each order [list obj (N_order,N_wav)]
#           - Ir:     Intensity values for each order [list of 2D arrays (N_order,N_obs,N_wav)]
#           - Blaze:  Blaze values for each order [list of 2D arrays (N_order,N_obs,N_wav)]
#           - Ia:     Telluric spectra for each order [list of 2D arrays (N_order,N_obs,N_wav)]
#           - T_obs:  Observation dates [BJD]
#           - phase:  Planet orbital phase - centered on mid-transit
#           - window: Transit window (1 --> mid-transit; 0 --> out-of-transit)
#           - berv:   BERV values [km/s]
#           - V0+Vp:  Stellar RV [km/s]
#           - airmass:Airmass values
#           - SN:     Signal-to-noise values for each order [N_order,N_obs]


savedata = (orders,WW,Ir,Bl,Ia,T_obs,phase,window,berv,V0+Vp,airmass,SN)
with open(name_fin, 'wb') as specfile:
    pickle.dump(savedata,specfile)
print("DONE")
