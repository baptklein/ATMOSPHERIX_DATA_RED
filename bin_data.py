#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 09:44:23 2023

@author: adminloc
"""

import numpy as np
import pickle

filename = "/home/adminloc/Bureau/Atmospheres/Data/Ups_And/read/UpsAnd_Oct24_inj10-H2O_broad.pkl"
# filename = "/home/adminloc/Bureau/Atmospheres/Data/AUMic/readfits/data_spirou_inject.pkl"

nbin = 4
nam_fin = "/home/adminloc/Bureau/Atmospheres/Data/Ups_And/read/UpsAnd_Oct24_inj10-H2O_broad_bin"+str(nbin)+".pkl"
with open(filename,'rb') as specfile:
    orders,WW,Ir,blaze,Ia,T_obs,phase,window,berv,vstar,airmass,SN = pickle.load(specfile)

nord     = len(orders)
nobs = len(T_obs)
nobs_bin = int(nobs/nbin)

T_final = np.zeros(nobs_bin)
phase_final = np.zeros(nobs_bin)
window_final = np.zeros(nobs_bin)
berv_final = np.zeros(nobs_bin)
vstar_final = np.zeros(nobs_bin)
airmass_final = np.zeros(nobs_bin)

for i in range(nobs_bin):
    T_final[i] = np.mean(T_obs[nbin*i:nbin*(i+1)])
    phase_final[i] = np.mean(phase[nbin*i:nbin*(i+1)])
    window_final[i] = np.mean(window[nbin*i:nbin*(i+1)])
    berv_final[i] = np.mean(berv[nbin*i:nbin*(i+1)])
    vstar_final[i] = np.mean(vstar[nbin*i:nbin*(i+1)])
    airmass_final[i] = np.mean(airmass[nbin*i:nbin*(i+1)])

I_tot=[]
blaze_tot= []
Ia_tot = []
SN_tot = []
for no in range(nord):
    I_ord = Ir[no]
    bl_ord = blaze[no]
    Ia_ord = Ia[no]
    SN_ord = SN[no]
    
    I_ord_final = np.zeros((nobs_bin,len(I_ord[0])))
    bl_ord_final = np.zeros((nobs_bin,len(bl_ord[0])))
    Ia_ord_final = np.zeros((nobs_bin,len(Ia_ord[0])))
    SN_ord_final = np.zeros(nobs_bin)

    for i in range(nobs_bin):
        I_ord_final[i] = np.mean(I_ord[nbin*i:nbin*(i+1)],axis=0)
        bl_ord_final[i] = np.mean(bl_ord[nbin*i:nbin*(i+1)],axis=0)
        Ia_ord_final[i] = np.mean(Ia_ord[nbin*i:nbin*(i+1)],axis=0)
        SN_ord_final[i] = np.mean(SN_ord[nbin*i:nbin*(i+1)])*np.sqrt(nbin)

    I_tot.append(I_ord_final)
    blaze_tot.append(bl_ord_final)
    Ia_tot.append(Ia_ord_final)
    SN_tot.append(SN_ord_final)


savedata = (orders,WW,I_tot,blaze_tot,Ia_tot,T_final,phase_final,window_final,berv_final,
            vstar_final,airmass_final,SN_tot)
with open(nam_fin, 'wb') as specfile:
    pickle.dump(savedata,specfile)