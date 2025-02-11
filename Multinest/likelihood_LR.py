import numpy as np
from numpy import ma

import sys
import os
import time

from scipy import interpolate

#import matplotlib.pyplot as plt
#import matplotlib

#from petitRADTRANS import nat_cst as nc


def return_like_LR(model,data):


    like = 0.0
    res = 400. 
    for i in range(len(data["data_LR_wavelength"])-1):
        #res  = data["data_LR_wavelength"][i]/(data["data_LR_wavelength"][i+1]-data["data_LR_wavelength"][i])
        lam = data["data_LR_wavelength"][i]
        lambda_array = np.linspace(lam-lam/res,lam+lam/res,15)
        model_i = np.average(model["interp_LR"](lambda_array))
        like -= (model_i-data["data_LR"][i])**2/(data["uncertainties_LR"][i])**2

#    res_spitzer= 10
#    for i in [len(data["data_LR_wavelength"])-2,len(data["data_LR_wavelength"])-2]:
#        lam = data["data_LR_wavelength"][i]
#        lambda_array = np.linspace(lam-lam/res_spitzer,lam+lam/res_spitzer,15)
#        model_i = np.average(model["interp_LR"](lambda_array))
#        like -= (model_i-data["data_LR"][i])**2/(data["uncertainties_LR"][i])**2


    return like/2

