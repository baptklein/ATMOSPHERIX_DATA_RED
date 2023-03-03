#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:44:37 2021

@author: florian
"""
import numpy as np
import pickle


c0          = 299792.458 

REP_DATA = "/home/florian/Bureau/Atmosphere_SPIRou/Data/Gl15A/HD189/reduced/"
filename = "Simu_GL15A_HD189_v30_1_proper_multi_onlyH2O-VMR3-T900_rotated3000-2_deletepaper_medianout_noboucher_nodetrend_autoPCA-1_imisttot.pkl"

dir_res = "/home/florian/Bureau/Atmosphere_SPIRou/Data/Gl15A/HD189/toMCMC/"
name_fin = "Simu_GL15A_HD189_v30_1_proper_multi_onlyH2O-VMR3-T900_rotated3000-2_deletepaper_medianout_noboucher_nodetrend_autoPCA-1_imisttot_MCMC.pkl"


with open(REP_DATA+filename,'rb') as ccfile:
    orders,W_data,I_data,T_obs,phase,window,berv,Vc,airmass,SN, proj = pickle.load(ccfile)


lambdas = np.zeros((2,130))

orders_final = []
I_final = []
V_data = []
Std_tot = []
Vstar = Vc-berv
Wmean_tot = []
proj_final = []

orders_to_select = orders

for k in range(len(orders)):
    if orders[k] in orders_to_select:
        orders_final.append(orders[k])
        
        Wmean = np.mean(W_data[k])
        Wmean_tot.append(Wmean)
        
        V_corr =c0*(W_data[k]/Wmean - 1.)
        V_data.append(V_corr)
        
        I_final.append(I_data[k])
    
        #We fit a second order parabola to the Std data
        Std = np.zeros(np.shape(I_data[k])[1])
        for i in range(np.shape(I_data[k])[1]):
            Std[i] = (np.std(I_data[k][:,i]))
        fit2 = np.polyfit(V_corr,Std,2)
        Stdfit = np.poly1d(fit2)
        Std_tot.append(Std)
        
        proj_final.append(proj[k])
        
        lambdas[:,orders[k]] = [W_data[k][0],W_data[k][-1]]
#    Std_tot.append(Stdfit(V_corr))
    
    else:
        continue
orders_final= np.array(orders_final)
savedata = (orders_final,Wmean_tot,V_data,I_final,Std_tot,phase,window,Vstar,proj_final)
with open(dir_res+name_fin, 'wb') as specfile:
    pickle.dump(savedata,specfile)    
    











