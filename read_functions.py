#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/2022 12:54:11 2021

@author: Baptiste & Florian
"""
import numpy as np
import os
from astropy.io import fits
from scipy.interpolate import PchipInterpolator
from global_parameters import c0, G

import batman


# -----------------------------------------------------------
# This function reads a time series of DRS-provided SPIRou files
# It stores some of the relevant information into "Order" objects
# and returns time series relevant for the analysis
# The spectra to read must be stored in a dedicated repository
# For the time being, the function can only read t.fits extensions 
# -----------------------------------------------------------
def read_data_spirou(repp,list_ord,nord):

    """
    --> Inputs:     - repp:      Path to the directory containing all the '.fits' files to read
                                 NOTE: files must be ordered in the chronologic order
                    - list_ord:  List of Order object
                    - nord:      Number of orders -- 49 for SPIRou

    --> Outputs:    - Attributes of Order objects:
                      1. W_raw (Wavelengths vectors)
                      2. I_raw (Time series of spectra)
                      3. blaze (Time series of blaze functions)
                      4. A_raw (Time series of telluric spectra computed from the DRS)
                      5. SNR (list of order S/N values for all observations)
                    - list_ord: upgraded list of orders
                    - airmass: airmass value for each observation
                    - bjd: time vector
                    - snr_mat: 2D matrix containing the S/N value for each observation and order (N_observation,N_order)
    """
    print(repp)
    nam_t     = sorted(os.listdir(repp))
    nobs      = len(nam_t)
    airmass   = np.zeros(nobs)
    bjd       = np.zeros(nobs)
    berv      = np.zeros(nobs)
    snr_mat   = np.zeros((nobs,nord))
    for nn in range(nobs):
        nmn          = repp + "/" + str(nam_t[nn])
        hdul_t       = fits.open(nmn)
        airmass[nn]  = float(hdul_t[0].header["AIRMASS"])
        bjd[nn]      = float(hdul_t[1].header["BJD"])
        berv[nn]     = float(hdul_t[1].header["BERV"])  
        i            = np.array(hdul_t[1].data,dtype=float) # intensity spectrum
        w            = np.array(hdul_t[2].data,dtype=float) # wavelength vector
        bla          = np.array(hdul_t[3].data,dtype=float) # blaze vector
        atm          = np.array(hdul_t[4].data,dtype=float) # telluric spectrum
        ### Get S/N values
        for mm in range(nord):
            num = 79 - list_ord[mm].number
            if num < 10: key = "EXTSN00" + str(num)
            else: key = "EXTSN0" + str(num)
            sn  = float(hdul_t[1].header[key]) # S/N for each order
            list_ord[mm].SNR.append(sn)
            snr_mat[nn,mm] = sn
        hdul_t.close()
        ## Store Order's attributes
        for mm in range(nord):
            O = list_ord[mm]
            num = 79 - list_ord[mm].number
            O.W_raw.append(w[num])
            O.I_raw.append(i[num])
            O.blaze.append(bla[num])
            O.I_atm.append(atm[num])
    for mm in range(nord):
        O       = list_ord[mm]
        O.SNR   = np.array(O.SNR,dtype=float)
        O.W_raw = np.array(O.W_raw,dtype=float)
        O.I_raw = np.array(O.I_raw,dtype=float)
        O.blaze = np.array(O.blaze,dtype=float)
        O.I_atm = np.array(O.I_atm,dtype=float)
    return list_ord,airmass,bjd,berv,snr_mat
 

    
# -----------------------------------------------------------
# Get transit window -- requires batman python module
# Uncomment lines below to use batman module to compute transit flux
# See information in https://lweb.cfa.harvard.edu/~lkreidberg/batman/
# -----------------------------------------------------------
def compute_transit(Rp,Rs,ip,T0,ap,Porb,ep,wp,limb_dark,uh,T_obs,ttype="primary",T_eclipse=0,fp=0):

    """
    --> Inputs:     - Rp:        Planetary radius
                    - Rs:        Stellar radius (same unit as Rp)
                    - ip:        Transit inclination [deg]
                    - T0:        Mid-transit time (same unit as T_obs -- here: bjd)
                    - ap:        Semi-major-axis [Stellar radius]
                    - Porb:      Planet orbital period (same unit as T_obs)
                    - ep:        Eccentricity of the planet orbit
                    - wp:        Argument of the periapsis for the planet orbit [deg] 
                    - limb_dark: Type of limb-darkening model: "linear", "quadratic", "nonlinear" see https://lweb.cfa.harvard.edu/~lkreidberg/batman/
                    - uh:        Limb-darkening coefficients matching the type of model and in the SPIRou band (Typically H or K)
                    - T_obs:     Time vector

    --> Outputs:    - flux:      Relative transit flux (1 outside transit) 
    """
#

    params           = batman.TransitParams()
    params.rp        = Rp/Rs                       
    params.inc       = ip
    params.t0        = T0
    params.a         = ap
    params.per       = Porb
    params.ecc       = ep
    params.w         = wp         
    params.limb_dark = limb_dark
    params.u         = uh
    params.fp        = fp
    params.t_secondary = T_eclipse
    if ttype=="secondary":
        bat              = batman.TransitModel(params,T_obs,transittype="secondary")
        flux             = bat.light_curve(params)
    else:
        bat              = batman.TransitModel(params,T_obs)
        flux             = bat.light_curve(params)
    return flux


# -----------------------------------------------------------
# Compute RV signature induced by the planet on its host star
# Assuming circular orbit for the planet
# -----------------------------------------------------------
def get_rvs(t,k,p,t0):

    """
    --> Inputs:     - t:   Time vector
                    - k:   Semi-amplitude of the planet-induced RV signal on the host star
                    - p:   Planet orbital period
                    - t0:  Planet mid-transit time

    --> Outputs:    - Planet-induced RV signature for the input time values 
    """

    return - k*np.sin(2.*np.pi/p * (t-t0))

# -----------------------------------------------------------
# Compute planet RV in the stellar rest frame
# -----------------------------------------------------------
def rvp(phase,k,v):
    """
    --> Inputs:     - phase: Planet orbital phase (T-T_obs)/Porb
                    - k:     Semi-amplitude of the planet RV
                    - v:     Planet RV at mid-transit

    --> Outputs:    - Planet RV for the input orbital phases
    """
    return k*np.sin(2.*np.pi*phase)+v
 
#### Main class -- Order
    
class Order:


    def __init__(self,numb):
        
        ### Generic information
        self.number       = numb    # Order number (in absolute unit -- 79: bluest; 31: reddest)
        self.W_mean       = 0.0     # Mean order wavelength
        self.SNR          = []      # DRS-computed S/N at the center of the order
        self.SNR_mes      = []      # Empirical estimate of the SNR before PCA
        self.SNR_mes_pca  = []      # Empirical estimate of the SNR after PCA

        ### Raw data information
        self.W_raw    = []      # Wavelength vectors for the raw observations - 2D matrix (time-wavelength)
        self.I_raw    = []      # Time series of observed spectra from the SPIRou DRS - 2D matrix (time-wavelength)                
        self.I_atm    = []      # Time series of Earth atmosphere spectra computed from the observations using Artigau+2014 method - DRS-provided 2D matrix (time-wavelength)
        self.blaze    = []      # Time series of blaze functions - 2D matrix (time-wavelength)

        ### Data reduction information
        self.W_fin    = []      # Final wavelength vector in the Geocentric frame
        self.W_bary   = []      # Final wavelength vector in the stellar rest frame
        self.I_fin    = []      # Reduced flux matrix before PCA cleaning
        self.I_pca    = []      # Reduced flux matrix after PCA cleaning
        self.proj     = []      # Reduced projected flux matrix after Gibson+21 cleaning



    # -----------------------------------------------------------
    # Pre-process of the DRS-provided spectra:
    # 1. Blaze normalization process
    # 2. Remove NaNs from each spectrum and convert sequences of
    #    spectra into np.array square matrices
    # -----------------------------------------------------------
    def remove_nan(self):
        
        """
        --> Inputs:      - Order object
        
        --> Outputs:     - Boolean: 1 --> order empty as NaNs everywhere; 0 otherwise 
        """

        ### Remove blaze
        I_bl = self.I_raw/self.blaze

        
        ### Spot the NaNs:
        ### In "*t.fits" files, regions of high telluric absorptions are replaced by NaNs
        ### as no precise estimation of the flux could be carried out
        ### Here we build a vector 'ind' stroring the position of the NaNs in every spectrum
        ind   = []
        for nn in range(len(I_bl)):
            i = np.where(np.isfinite(I_bl[nn])==True)[0]
            ind.append(i)
        r  = np.array(list(set.intersection(*map(set,ind))),dtype=int)
        r  = np.sort(np.unique(r))

        ### remove the NaNs
        I_ini = []
        W_ini = []
        B_ini = []
        A_ini = []
        for nn in range(len(I_bl)):
            I_ini.append(I_bl[nn,r])
            W_ini.append(self.W_raw[nn,r])
            A_ini.append(self.I_atm[nn,r])
            B_ini.append(self.blaze[nn,r])

        ### Convert into 2D array object
        self.I_raw  = np.array(I_ini,dtype=float)    
        self.W_raw  = np.array(W_ini,dtype=float)[0]   
        self.I_atm  = np.array(A_ini,dtype=float)
        self.B_raw  = np.array(B_ini,dtype=float) 
        self.W_mean = self.W_raw.mean()   ### Compute mean of the actual observations

        ### Remove the order if it contains only NaNs
        if len(self.I_raw[0]) == 0:
            tx = "\nOrder " + str(self.number) + " is empty and thus removed from the analysis"
            print(tx)
            return 1
        else:
            return 0




    # -----------------------------------------------------------
    # Add a synthetic planet signature to an input sequence of spectra
    # 1. Compute a synthetic sequence of spectra by:
    #    (i)  shifting a planetary template (normalized transit depth)
    #         according to the planet RV
    #    (ii) Weight by a transit window (1 at mid-transit, 0 outside
    #         of transit
    # 2. Interpolate each synthetic planet spectrum and multiply an
    #    input sequence of spectra by the synthetic sequence of transit
    #    depths
    # -----------------------------------------------------------
    def add_planet(self,type_obs,Wm,Im,window,planet_speed,Vc,ampl=1.0,pixel=np.linspace(-1.13,1.13,11)):

        """
        --> Inputs:      - Order object
                         - type_obs (str) emission or tranmission
                         - Wm:      Wavelength vector of the planet atmosphere template
                         - Im:      Template of wavelength-dependent transit depth (i.e., model) 
                         - window:  Transit window
                         - planet_speed : speed of the planet along the orbit
                         - Vc:      Velocimetric correction to move from Geocentric frame to stellar rest frame
                                    Typically: Vc = Stellar systemic vel. + planet-signature RV - Barycentric Earth RV 
                         - ampl:    Amplification factor: amplify the injected planetary signal

        --> Outputs:     - self.I_raw_pl
        """
        self.Wm    = Wm
        self.Im    = ampl*(Im-np.max(Im))+np.max(Im)        
        self.I_syn = np.zeros(self.I_raw.shape)

        Imm  = self.Im#/np.min(flux)
        if type_obs =="transmission":
            tdepth_interp     = PchipInterpolator(self.Wm,Imm)  # Interpolate model
        else:
            flux_interp = PchipInterpolator(self.Wm,Imm)  
            
        if type_obs =="transmission":
            for nn in range(len(self.I_raw)): # For each observation date
                if window[nn] != 0.0:
                    I_ttt = np.zeros(len(self.W_raw))
                    
                    # Shift model in the Geocentric frame
                    for pp in pixel: I_ttt += tdepth_interp(self.W_raw/(1.0+((planet_speed[nn]+Vc[nn]+pp)/(c0/1000.))))
                    self.I_syn[nn] = I_ttt/len(pixel)*window[nn]
                
        else:
            for nn in range(len(self.I_raw)): # For each observation date
                I_ttt = np.zeros(len(self.W_raw))
                # Shift model in the Geocentric frame
                for pp in pixel: I_ttt += flux_interp(self.W_raw/(1.0+((planet_speed[nn]+Vc[nn]+pp)/(c0/1000.))))
                self.I_syn[nn]  = I_ttt/len(pixel)
                
                

 

