#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:48:34 2023

@author: florian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/2022 12:54:11 2021

@author: Baptiste & Florian
"""
import numpy as np


pipeline_rep = "/home/florian/Bureau/Atmosphere_SPIRou/Pipeline_git/"

### Name of the picle file to read the data from
filename = pipeline_rep+'test.pkl'
#output file
nam_fin  = pipeline_rep+'test_new.pkl' 
#information file
nam_info = pipeline_rep+"info.dat"


### Correction of stellar contamination
### Only used if synthetic spectrum available
corr_star  = False
WC_name    = ""            ### Input wavelength for synthetic stellar spectra
IC_name    = ""            ### Input flux for synthetic stellar spectra

### Additional Boucher correction. If dep_min >=1, not included. 
dep_min  = 0.7    # remove all data when telluric relative absorption > 1 - dep_min
thres_up = 0.05      # Remove the line until reaching 1-thres_up
Npt_lim  = 800      # If the order contains less than Npt_lim points, it is discarded from the analysis

### Interpolation parameters
pixel    = np.linspace(-1.14,1.14,11)   ### Sampling a SPIRou pixel in velocity space -- Width ~ 2.28 km/s
sig_g    = 2.28                       ### STD of one SPIRou px in km/s
N_bor    = 15                           ### Nb of pts removed at each extremity (twice)

### Normalisation parameters
N_med    = 150                          ### Nb of points used in the median filter for the inteprolation
sig_out  = 5.0                          ### Threshold for outliers identification during normalisation process 
N_adj = 2 ### Number of adjacent pixel removed with outliers
deg_px   = 2                            ### Degree of the polynomial fit to the distribution of pixel STDs

### Parameters for detrending with airmass
det_airmass = False
deg_airmass = 2

### Parameters PCA. Auto-tune automatically decides the number of component 
#to remove by comparing with white noise map.
mode_pca    = "pca"                     ### "pca" or "autoencoder"
wpca = False   #Use weighted pca
auto_tune   = True                  ### Automatic tuning of number of components
factor_pca = 1. #factor in the auto tune: every PC above factor*white_noise_mean_eigenvalue is suppressed
mode_norm_pca = "none" #how to remove mean and std in the data before PCA. Four possibilities:
                         # "none" : data untouched.
                         # "global" : suppression of mean and division by the std of the whole data set 
                         # 'per_pix': same as global but column by colum (per pixel)
                         # 'per_obs': same as global but line by line (per observation)

 ### Nb of removed components if auto tune is false
npca        = np.array(1*np.ones(49),dtype=int)     


### Plot info
plot_red    = True
numb        = 46
nam_fig     = pipeline_rep+"Figures/reduc_" + str(numb) + "_jun19.png"
    

#If you want to remove some orders, put them here
orders_rem     = []


#We can manually decide the where is the transit in the phase direction,
#and exclude it for the calculation of the mean stellar spectrum.
#If set_window = False, the transit window defines n_ini and n_end
set_window = False
n_ini_fix,n_end_fix = 10,20    ### Get transits start and end indices

### Size of the estimation of the std of the order for final metrics
N_px          = 200




