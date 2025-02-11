import numpy as np
from numpy import ma

import sys
import os
import time

from scipy import interpolate


import model_interpolate_functions as modfunc


def prepare_to_likelihood(config,model,data,param):

    
    
    mod_2D = []
    for mod in model["models"]:
        mod_dic = dict(nord=mod.ord,wavelength=mod.Wm,absorption=mod.Rp)
        mod_2D_order = modfunc.Model_2D(mod_dic)
        mod_2D_order.create_interp()
        mod_2D.append(mod_2D_order)
  
	# function of interpolation of the model at each phase
    
    # for mod in mod_2D:
    #     mod.build_model(P.window)

    sig_v_int = 1.15   ### Half-width of the window [km/s] (we take half of SPIRou pixel)
    N_v_int   = 7     ### Number of points of the integration
    ddv = np.linspace(-sig_v_int,sig_v_int,N_v_int)
    
    final_model = []
    final_data = []
    final_std = []
    final_number = []
    
    for i in range(config["num_transit"]):
        
        #we only select window>0.2
        start = np.where(data["window"][i]>0.2)[0][0] 
        end =  np.where(data["window"][i]>0.2)[0][-1]+1 

        
        tot_indiv = modfunc.total_model(param["Kp"],param["Vsys"],mod_2D,
                                        data["orders"][i],
                                        data["wmean"][i],
                                        data["V"][i],data["phase"][i],data["window"][i],data["Vstar"][i],ddv,data["proj"][i])
        tot_indiv.fill_model()
        indiv_model= tot_indiv.bin_model()
        indiv_data =[]
        indiv_std = []
        number = 0
        for j in range(len(data["orders"][i])):
            #print(data["intensity"][i][j][start])
            indiv_data= indiv_data+(((data["intensity"][i][j][:,50:-50]).T-np.mean((data["intensity"][i][j][:,50:-50]).T,axis=0)).T)[start:end].tolist() #The list format allows not to care about shapes
            indiv_std = indiv_std+(np.asarray([data["std"][i][j][50:-50]]*(end-start))).tolist() #duplicate Std end-start times
            #print(len(indiv_data))
            number+=end-start
        final_model = final_model+indiv_model # concatenate the models to have a list of spectra
        final_data = final_data+indiv_data
        final_std = final_std+indiv_std# same here for the data    
        final_number.append(number)
    return {
			"data": final_data,
            "model": final_model,
            "std" : final_std,
            "number" : final_number
        }
