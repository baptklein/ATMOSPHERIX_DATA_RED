#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:56:24 2022

@author: florian
"""

import numpy as np # linear algebra
import random as rad
import os
import gc 
import seaborn 
from matplotlib import pyplot as plt
from pylab import *
import time

import torch
from torch import tensor
from torch import nn

from tqdm.notebook import tqdm as tqdm
import pickle
from encoder_functions import *

def apply_encoder(flux):
    
    # #The data are supposed to be airmass corrected and bad pixels filtered. For one order an associated wavelength and time array is recommended for meaningful plots.
    # #The flux is switched in the log space.

    flux = flux+1.0  #to avoid log issues

    flux = np.log(flux)
    im = np.nanmean(flux)
    ist = np.nanstd(flux)
    flux = (flux - im)/ist
    
    # ##### /!\ To be able to use CUDA and have a feasible computation time you need access to a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    
    train_x, train_y, test_x, test_y = get_training_and_test_sets(flux)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    
    ######################### uncomment here to test/debug the encoder

    # start_time = time.time()

    # model, train_loss, test_loss = train_model(train_x, train_y, test_x, test_y,device,debug=True)
    # print("--- %s seconds ---" % (time.time() - start_time))

    # train_loss = [x.detach().cpu().numpy() for x in train_loss]
    # test_loss = [y for y in test_loss]

    # start = 0
    # end = -1
    # fig, ax = plt.subplots(figsize=(18,6))
    
    # ax.plot(range(len(train_loss))[start:end], train_loss[start:end], 'k-', label='training set')
    # ax.plot(range(len(test_loss))[start:end], test_loss[start:end], 'r-', label='test set')
    # ax.legend(fontsize=15)
    # ax.tick_params(labelsize=15)
    # ax.set_xlabel('Epochs', size=18, weight='bold')
    # ax.set_ylabel('Loss', size=18, weight='bold')
    # stop
    ##########################

    model, _, _ = train_model(train_x, train_y, test_x, test_y,device)

    x_aec = tensor(np.float32(flux))
    x_aec = x_aec.to(device)
    output = (model(tensor(x_aec).float().to(device)).detach().cpu().numpy())

    return(np.exp((flux-output)*ist+im)-1.0)