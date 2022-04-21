import numpy as np
from numpy import ma

import sys
import os
import time

from scipy import interpolate

#import matplotlib.pyplot as plt
#import matplotlib

import model_interpolate_functions as modfunc
from petitRADTRANS import nat_cst as nc


def prepare_to_likelihood(config,model,data,param):

    
    
    mod_2D = []
    for mod in model["models"]:
        mod_dic = dict(nord=mod.ord,wavelength=mod.Wm,absorption=mod.Rp)
        mod_2D_order = modfunc.Model_2D(mod_dic)
        mod_2D_order.create_interp()
        mod_2D.append(mod_2D_order)
     
        
    Rp_Rs = config["radius_RJ"]*nc.r_jup_mean/(config["Rs_Rsun"]*7.0e10)


	# function of interpolation of the model at each phase
    
    # for mod in mod_2D:
    #     mod.build_model(P.window)

    sig_v_int = 1.15   ### Half-width of the window [km/s] (we take half of SPIRou pixel)
    N_v_int   = 7     ### Number of points of the integration
    ddv = np.linspace(-sig_v_int,sig_v_int,N_v_int)
    
    final_model = []
    final_data = []
    final_std = []
    
    for i in range(config["num_transit"]):
        
        start = np.where(data["window"][i]>0.2)[0][0] 
        end =  np.where(data["window"][i]>0.2)[0][-1]+1 ## limits where the window is >0.2

        
        tot_indiv = modfunc.total_model(param["Kp"],param["Vsys"],mod_2D,
                                        data["orders"][i],
                                        data["wmean"][i],
                                        data["V"][i],data["phase"][i],data["window"][i],data["Vstar"][i],ddv)
        tot_indiv.fill_model()
        indiv_model= tot_indiv.bin_model()
        indiv_data =[]
        indiv_std = []
        for j in range(len(data["orders"][i])):
            #print(data["intensity"][i][j][start])
            indiv_data= indiv_data+(data["intensity"][i][j]-np.mean(data["intensity"][i][j],axis=0))[start:end].tolist() #The list format allows not to care about shapes
            indiv_std = indiv_std+(np.asarray([data["std"][i][j]]*(end-start))).tolist() #duplicate Std end-start times
            #print(len(indiv_data))
        final_model = final_model+indiv_model # concatenate the models to have a list of spectra
        final_data = final_data+indiv_data
        final_std = final_std+indiv_std# same here for the data    
    return {
			"data": final_data,
            "model": final_model,
            "std" : final_std
        }
