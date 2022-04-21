import numpy as np
from numpy import ma

import sys
import os
import time

from scipy import interpolate
import batman

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

	### Enter the planet parameters
    P = []
    for i in range(config["num_transit"]):
        P_indiv =modfunc.Planet(Rp_Rs,config,config["dates"][i])
	### Generate the transit event
        P_indiv.make_batman()

	### Build weighting window to transform the template of 2D sequence
        P_indiv.make_window()
        P.append(P_indiv)


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
        
        start = P[i].start_transit
        end =  P[i].end_transit+1 ## limits where the window is >0.2

        
        tot_indiv = modfunc.total_model(param["Kp"],param["Vsys"],
                                        config["orders"][i],
                                        config["Wmean"][i],
                                        mod_2D,data["V"][i],P[i],ddv)
        tot_indiv.fill_model()
        indiv_model= tot_indiv.bin_model()
        indiv_data = []
        indiv_std = []
        for j in range(len(config["orders"][i])):
            #print(data["intensity"][i][j][start])
            indiv_data= indiv_data+data["intensity"][i][j][start:end]
            indiv_std = indiv_std+data["std"][i][j][start:end]
            #print(len(indiv_data))
        final_model = final_model+indiv_model # concatenate the models to have a list of spectra
        final_data = final_data+indiv_data
        final_std = final_std+indiv_std# same here for the data    
    return {
			"data": final_data,
            "model": final_model,
            "std" : final_std
        }
