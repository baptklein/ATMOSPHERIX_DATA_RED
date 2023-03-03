# -*- coding: utf-8 -*-
"""
Created on Sep 2021
@authors: Baptiste KLEIN
"""

# Useful libraries
import numpy as np
import os
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from astropy.modeling import models, fitting, polynomial
from astropy.stats import sigma_clip
from scipy.optimize import minimize
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import matplotlib
from plots import *

class Constants:
    
    def __init__(self):
        self.c0 = 299792.458 #km/s



# -----------------------------------------------------------
# This class stores information about each SPIRou orders
# and contains methods to remove residuals of stellar and
# Earth atmosphere spectra
# -----------------------------------------------------------

class Order:


    def __init__(self,numb):
        
        ### Generic information
        self.number   = numb    # Order number (in absolute unit -- 79: bluest; 31: reddest)
        self.W_mean   = 0.0     # Mean order wavelength
        self.SNR      = []      # DRS-computed S/N at the center of the order
        self.SNR_mes  = []      # Empirically-computed S/N at the center of the order after data reduction

        ### Planet template information
        self.W_mod    = []      # Wavelength vector of the planetary template (very high resolution)
        self.I_mod    = []      # Planet template (wavelength-dependent transit depth)
        self.I_syn    = []      # Synthetic planet signature (2D sequence of spectra in time-wavelength domain)

        ### Raw data information
        self.W_raw    = []      # Wavelength vectors for the raw observations - 2D matrix (time-wavelength)
        self.I_raw    = []      # Time series of observed spectra from the SPIRou DRS - 2D matrix (time-wavelength)
        self.I_raw_jup= []      # Time series of observed spectra + injected planet signature - 2D matrix (time-wavelength)
        self.I_raw_nep= []      # Time series of observed spectra + injected planet signature - 2D matrix (time-wavelength)
                
        self.I_atm    = []      # Time series of Earth atmosphere spectra computed from the observations using Artigau+2014 method - DRS-provided 2D matrix (time-wavelength)
        self.blaze    = []      # Time series of blaze functions - 2D matrix (time-wavelength)
        self.B_raw    = []

        ### Data reduction information
        self.W_cl     = []      # Wavelength vector after removing regions of high telluric absorption following Boucher+2021 (1D vector) -- Step 1.1
        self.I_cl     = []      # Time series of spectra after removing regions of high telluric absorption (2D matrix) -- Step 1.1
        self.W_al     = []      # Wavelength vector after aligning spectra in the stellar rest frame (same as self.W_cl but edges are cut) -- Step 1.2
        self.I_al     = []      # Time series of spectra in the stellar rest frame (2D matrix - each order extremity is cut) -- Step 1.2
        self.I_sub    = []      # Time series of spectra after removing reference spectrum -- Step 1.3
        self.I_norm   = []      # Time series of normalized spectra (+outlier cleaning) -- Step 1.4
        self.W_norm   = []      # Wavelength vector corresponding to the normalized spectra -- Step 1.4
        self.I_det    = []      # Time series of spectra after detrending with airmass -- step 1.5
        self.I_red    = []      # Time series of spectra after bad pixel cleaning but before applying PCA -- step 1.6
        self.W_red    = []      # Wavelength vector corresponding to self.I_red -- step 1.6
        self.I_pca    = []      # Time series of spectra after PCA reduction -- step 1.7




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
    def add_planet(self,Wm,Im,window,phase,K_inj,V_inj,Vc,ampl=1.0):

        """
        --> Inputs:      - Order object
                         - Wm:      Wavelength vector of the planet atmosphere template
                         - Im:      Template of wavelength-dependent transit depth (i.e., model) 
                         - window:  Transit window
                         - phase:   Vector of orbit phase
                         - K_inj:   Semi-amplitude of the injected planet signature [km/s]
                         - V_inj:   Planet radial velocity at mid-transit [km/s]
                         - Vc:      Velocimetric correction to move from Geocentric frame to stellar rest frame
                                    Typically: Vc = Stellar systemic vel. + planet-signature RV - Barycentric Earth RV 
                         - ampl:    Amplification factor: amplify the injected planetary signal

        --> Outputs:     - self.I_raw_pl
        """
        self.Wm    = Wm
        self.Im    = ampl*(Im-np.max(Im))+np.max(Im)
        vp         = rvp(phase,K_inj,V_inj)     # Compute planet RV
        self.I_syn = np.zeros(self.I_raw.shape)
        pixel      = np.linspace(-1.13,1.13,11) # Window for integrating over a SPIRou pixel [km/s] -- Velocity bins of 2.28 km/s (Donati+2020b)


        for nn in range(len(self.I_raw)): # For each observation date
        
            Imm  = self.Im#/np.min(flux)
        
            if window[nn] != 0.0:
                f     = interp1d(self.Wm,Imm,kind="linear")  # Interpolate model
                I_ttt = np.zeros(len(self.W_raw))
                
                # Shift model in the Geocentric frame
                for pp in pixel: I_ttt += f(self.W_raw/(1.0+((Vc[nn]+vp[nn]+pp)/Constants().c0)))
                self.I_syn[nn] = I_ttt/len(pixel)*window[nn]
                
        self.I_syn=1-self.I_syn
        I_fin = self.I_raw*self.I_syn
        return self.I_syn
        
        #self.I_raw_pl = self.I_raw*self.I_syn # Add model to the observations







    # -----------------------------------------------------------
    # Remove regions of strong telluric absorption
    # 1. From DRS-provided telluric spectrum, spot all regions
    #    with more than 'dep_min' relative absorption depth.
    # 2. For all regions identified, remove points on both sides
    #    until reaching thres_up relative absorption level wrt
    #    the continuum
    # -----------------------------------------------------------
    def remove_tellurics(self,dep_min,thres_up):

        """
        --> Inputs:     - Order object
                        - dep_min: Threshold (in relative absorption unit) above
                                   which lines are removed
                        - thres_up: When removing a region of strong absorption,
                                    all points are remove until reaching a relative
                                    absorption of 'thres_up'

        --> Outputs:    - self.I_cl; self.W_cl  
        """
        
        ### Identify regions of strong telluric absorption from the median DRS-provided
        ### telluric spectrum
        Am      = np.median(self.I_atm,axis=0)
        ind_A   = np.where(Am<1-dep_min)[0]

        ### Identify regions adjacent to the telluric absorption lines spotted in the previous step
        ind_tel = []
        for ii in range(len(ind_A)):
            i0     = ind_A[ii]
            n_ini2 = 0
            while Am[i0-n_ini2] < 1 - thres_up:
                if i0 - n_ini2 == 0: break
                n_ini2 += 1
            n_end2 = 0
            while Am[i0+n_end2] < 1 - thres_up:
                if i0 + n_end2 == len(Am)-1: break
                n_end2 += 1
            itel = np.arange(i0-n_ini2,i0+n_end2+1)  
            ind_tel.append(itel)
        if len(ind_tel)>0: ind_tel = np.sort(np.unique(np.concatenate(ind_tel)))
        else:              ind_tel = []

        ### Remove regions of strong telluric absorption
        self.I_cl    = np.delete(self.I_raw,ind_tel,axis=1)
        self.W_cl    = np.delete(self.W_raw,ind_tel)



    # -----------------------------------------------------------
    # Align spectra in the stellar rest frame
    # 1. Interpolate spectra using interp1d scipy method
    # 2. Shift the spectrum given a user-input velocity correction
    # WARNING -- From observer (geocentric) to stellar rest frame only
    # -----------------------------------------------------------
    def align(self,Vc,kind="linear",n_bor=30):

        """
        --> Inputs:     - Order object
                        - Vc: Geocentric-to-stellar rest frame velocity correction [km/s]
                              Typically: Vc = Stellar systemic vel. + planet-signature RV - Barycentric Earth RV
                        - kind: kind of interpolation matching the scipy interp1d documentation:
                                https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
                        - n_bor: Number of points removed at each edge of the order (to avoid extrapolation)

        --> Outputs:    - self.I_al, self.W_al
        """

        c0      = Constants().c0
        I_al    = np.zeros(self.I_cl.shape)
        pixel   = np.linspace(-1.13,1.13,11)
        for nn in range(len(self.I_cl)):
            fi        = interp1d(self.W_cl,self.I_cl[nn],kind=kind,fill_value="extrapolate")
            I_tmp     = np.zeros(len(self.W_cl))
            for pp in pixel: I_tmp    += fi(self.W_cl*(1.0+((Vc[nn]+pp)/c0)))
            I_al[nn]      = I_tmp/len(pixel)
        self.I_al      = I_al[:,int(n_bor):-int(n_bor)]
        self.W_al      = self.W_cl[int(n_bor):-int(n_bor)]






    # -----------------------------------------------------------
    # Subtract a reference spectrum from the observed sequence of spectra
    # -- Step 1.3 of the data reduction procedure
    # 1. Compute reference spectrum (median of all spectra or out-of-transit spectra)
    # 2. Linear fit of the reference spectrum to each observed spectrum
    # 3. Division by best-fitting spectrum
    # -----------------------------------------------------------
    def subtract_stellar(self,mode="full",n_ini=0,n_end=0,plot=False):

        """
        --> Inputs:     - Order object
                        - mode: "full" --> compute median spectrum from all observed spectra
                                "out"  --> compute median spectrum from out-of-transit spectra only
                        - n_ini: index of the last spectrum taken before the transit
                        - n_end: index of the first spectrum taken after the transit
                        - plot:  Boolean --> if True, plot the best-fitting solution for a number nb of spectra

        --> Outputs:    - self.I_sub
                        - if plot == True --> return figure
        """
    
        ### Compute reference spectrum
        if mode  == "out":
            I_out      =  np.concatenate((self.I_al[:n_ini,:],self.I_al[n_end:,:]),axis=0)
        else: I_out    = self.I_al
        I_ref          = np.median(I_out,axis=0)

        ### Linear fit to each observed spectrum
        basis    = np.ones((2,len(self.W_al)))
        basis[1] = I_ref
        I_pred_f = np.zeros(np.shape(self.I_al))
        I_sub    = np.zeros(np.shape(self.I_al))
        c_fin    = np.zeros((len(self.I_al),2))
        for nn in range(len(self.I_al)):
            coeffs       = lstsq(basis.T,self.I_al[nn],rcond=None)[0]
            I_pred       = np.dot(coeffs,basis)
            I_pp         = self.I_al[nn]/I_pred 
            I_pred_f[nn] = I_pred
            c_fin[nn]    = coeffs
            I_sub[nn]    = I_pp
        self.I_sub       = I_sub
        
        ### Change eps --> space between consecutive spectra
        ### Change nb  --> Nb of spectra shown on the plot
        if plot == True: plot_spec_fit(self.W_al,self.I_al,I_pred_f,eps=0.03,nb=10)
            
        ### Uncomment the line below to plot only residuals of the fit
        #if plot == True: plot_spec_fit(self.W_al,self.I_al/I_pred_f,[],eps=0.03,nb=10)








    # -----------------------------------------------------------
    # Normalize and remove outlier for each residual spectrum
    # -- Step 1.4 of the data reduction procedure
    # Iterative process: For each spectrum I
    # 1. Apply median filter and normalize I by the smoothed spectrum
    # 2. Remove outliers (+a few adjacent points)
    # 3. Repeat the process until no outlier identified
    # 4. Interpolate the spectrum after outlier removal and replace
    #    location of former outliers by averaged of closest non-NaN points
    # 5. Re-apply median filter and normalize the spectrum
    # -----------------------------------------------------------
    def normalize(self,N_med,sig_out,N_adj=2,nbor=30,plot=False):

        """
        --> Inputs:     - Order object
                        - N_med:   Cut-off frequency of the median filter (nb of points of the sliding window of
                                   the moving median)
                        - sig_out: Threshold for outlier removal (in sigma)
                        - N_adj:   Number of adjacent points to each outlier removed with the outlier 
                        - nbor:    Number of points removed at each edge of the order
                        - plot:    Boolean --> if True, plot the normalization solutuon for a number nb of spectra

        --> Outputs:    - self.I_norm, self. W_norm
                        - if plot == True --> return figure
        """

        ### Initialization
        Is          = self.I_sub
        Ws          = self.W_al
        ind_out_fin = []
        self.I_norm = np.zeros((len(self.I_sub),len(self.W_al)-int(2*nbor)))
        I_corr      = np.zeros(self.I_sub.shape)
        I_med_tot   = np.zeros(self.I_sub.shape)
        
        for nn in range(len(Is)):
            I           = np.copy(Is[nn]) # Temp. spectrum
            W           = np.copy(Ws)
            out         = True  ## Is there outlier?
            ind_out     = []
            
            while out == True:
                
                #### Apply median filter
                I_med = median_filter(I,N_med)       # Median filter
                In    = I/I_med                      # Normalize temp. spectrum
                S     = sigma_clip(In,sigma=sig_out) # Identify outlier 
                indo  = np.where(S.mask)[0]          # Spot outlier locations
                
                ### Remove points adjacent to identified outlier on both sides of each identified outlier
                ind = []
                for jj in indo:
                    if jj-N_adj < 0: ini = 0
                    else: ini = jj-N_adj
                    if jj + N_adj + 1 > len(W): end = len(W)
                    else: end = jj + N_adj +1
                    ll = np.arange(ini,end)
                    ind.append(ll)
                if len(ind) > 0:
                    ind_out.append(np.unique(np.concatenate(ind)))
                    I = np.delete(I,np.unique(np.concatenate(ind)))
                    W = np.delete(W,np.unique(np.concatenate(ind)))
                else: out = False
                    
            ### Interpolate over outliers and apply median filter
            f               = interp1d(W,I,"linear",fill_value="extrapolate")
            If              = f(Ws)
            Im              = median_filter(If,N_med)
            In              = If/Im
            self.I_norm[nn] = In[nbor:-nbor] # Remove pts at both edge of the order

            if plot == True:
                I_corr[nn]    = If
                I_med_tot[nn] = Im
        self.W_norm = self.W_al[nbor:-nbor]  

        ### Change eps --> space between consecutive spectra
        ### Change nb  --> Nb of spectra shown on the plot
        if plot == True: plot_normalization(Ws,Is,I_corr,I_med_tot,eps=0.04,nb=5)





    # -----------------------------------------------------------
    # Detrending with airmass
    # -- Step 1.5 of the data reduction procedure
    #  Fit a polynomial model of airmass to the sequence of normalized
    #  spectra
    # -----------------------------------------------------------
    def detrend_airmass(self,airmass,deg=2):

        """
        --> Inputs:     - Order object
                        - airmass: vector of airmass
                        - deg:     Degree of the airmass model (2 in generally sufficient)


        --> Outputs:    - self.I_det
                        - pb: vector of best-fitting spectra
                        - I_pred: Best-fitting modelled sequence of spectra
        """
        
        indw    = np.argmin(np.abs(self.W_norm-self.W_mean))
        Il      = np.log(self.I_norm)
        COV_inv = np.diag(1./np.std(Il[:,indw-200:indw+200],axis=1)**(2)) ## Assumes that normalized spectra dominated by white noise
        X       = np.ones((len(self.I_norm),deg+1))
        for ii in range(deg): X[:,ii+1] = airmass**(ii+1)
        pb,pbe = LS(X,COV_inv,Il) ## Least-squares estimator
        I_pred = np.dot(X,pb)
        self.I_det = np.exp(Il - I_pred)
        return pb, I_pred




    # -----------------------------------------------------------
    # Filter bad pixels
    # -- Step 1.6 of the data reduction procedure
    #  1. Compute standard deviation for each pixel
    #  2. Fit a parabola to the distribution of pixel dispersion
    #  3. Remove all outliers above fit
    # -----------------------------------------------------------
    def filter_pixel(self,deg_px=2,sig_px=4.,plot=False,verbose=True):

        """
        --> Inputs:     - Order object
                        - deg_px:  degree of the polynomial fit (2 is generally sufficient)
                        - sig_px:  Threshold for outlier removal
                        - plot:    Boolean --> if True, then plot distribution of pixel dispersions
                                   and best-fitting polynomial + indicate identified outliers 
                        - verbose: Boolean --> if True, indicate the nb of outliers removed for the order

        --> Outputs:    - self.I_red, self.W_red
                        - if plot == True, then return figure
        """

        n_iter_fit    = 10 ### Number of iterations for the iterative polynomial fit with outliers removal
                           ### See documentation on astropy --> https://docs.astropy.org/en/stable/api/astropy.modeling.fitting.FittingWithOutlierRemoval.html
        rms_px        = self.I_det.std(axis=0) # Dispersion of each pixel (computed along the time axis)
        WW            = self.W_norm - np.mean(self.W_norm) # Wavelength vector for the fit
        model,filt_px = poly_fit(WW,rms_px,deg_px,sig_px,n_iter_fit) ### See functions below
        rms_pred      = model(WW) # Best prediction

        ### Identify and remove outliers
        ind_px        = []
        for ii in range(len(filt_px)):
            if filt_px[ii] != "--": ind_px.append(ii)
            elif rms_px[ii] < rms_pred[ii]: ind_px.append(ii)
        self.I_red = self.I_det[:,ind_px]
        self.W_red = self.W_norm[ind_px]

        ### If verbose --> print nb of pixels discarded
        if verbose: print("Order",self.number,":",len(WW)-len(ind_px),"pixel removed")

        ### If plot --> plot best-fitting solutions and removed pixels
        if plot: plot_distrib(self.W_norm,rms_px,rms_pred,self.W_red,rms_px[ind_px],self.number)




    # -----------------------------------------------------------
    # Compute S/N of the resulting sequence of spectra
    # For each spectrum, compute the dispersion on ~800 pixels at the center of the order
    # -- Step 1.6 of the data reduction process
    # -----------------------------------------------------------
    def get_pixel_dispersion(self,N_px):

        """
        --> Inputs:     - Order object
                        - N_px: Nb of pixels used to compute the empirical estimate of the S/N

        --> Outputs:    - self.SNR_mes
        """


        ind          = np.argmin(np.abs(self.W_red-self.W_red.mean())) ## Spot order center
        It           = self.I_red[:,ind-N_px:ind+N_px]
        self.SNR_mes = 1./np.std(It,axis=1) ## Compute empirical S/N
        print(self.SNR_mes)





# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTIONS FOR THE DATA REDUCTION PROCESS




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
import batman
def compute_transit(Rp,Rs,ip,T0,ap,Porb,ep,wp,limb_dark,uh,T_obs):

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
    bat              = batman.TransitModel(params,T_obs)
    flux             = bat.light_curve(params)
    return flux



# -----------------------------------------------------------
# Get transit initial and end dates from an input transit curve
# -----------------------------------------------------------
def get_transit_dates(flux):

    """
    --> Inputs:     - flux:  Relative transit flux (e.g., output from batman python module)

    --> Outputs:    - n_ini: Index of the last point before transit
                    - n_end: Index of the first point after transit
    """

    n_ini,n_end = 1,1
    if flux[0] < 1.0: n_ini = 0
    else: 
        cc = 0
        while flux[cc] == 1.0:
            cc += 1
        n_ini = cc-1
    if flux[-1] < 1.0: n_end = 0
    else:
        cc = n_ini + 1
        while flux[cc] < 1.0:
            cc += 1
        n_end = cc
    return n_ini,n_end



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


# -----------------------------------------------------------
# Simple hyperbola
# -----------------------------------------------------------
def hyp(par,xx):
    return par[0]/xx + par[1]

# -----------------------------------------------------------
# Simple inverse hyperbola
# -----------------------------------------------------------
def hyp_inv(par,yy):
    return par[0]/(yy-par[1])

# -----------------------------------------------------------
# Return least-square difference between a hyperbola for 'par' 
# parameters and data yy.
# xx is the X-axis vector 
# -----------------------------------------------------------
def crit_hyp(par,xx,yy):
    y_pred = hyp(par,xx)
    return np.sum((yy-y_pred)**(2))  


# -----------------------------------------------------------
# Compute Order to mean wavelength equivalence 
# Usage: Plot order number as X-axis and mean wavelengths as Y axis
# In practice: fits an hyperbola between order nb and mean wavelength
# See function plots.plot_orders for more information
# -----------------------------------------------------------
def fit_order_wave(LO,wm_fin):

    """
    --> Inputs:     - LO: list of order numbers
                    - wm_fin: list of the mean wavelengths corresponding to LO

    --> Outputs:    - WW: Wavelength ticks for the plot
                    - LO_pred: order numbers corresponding to WW
                    - LO_predt: densely-sampled list of orders for minor ticks locators
    """

    par0    = np.array([100000,200.0],dtype=float) 
    res     = minimize(crit_hyp,par0,args=(LO,wm_fin))
    p_best  = res.x 
    LO_tot  = np.arange(29,81)
    pp      = hyp(p_best,LO_tot)
    WWT      = np.linspace(2400,900,16)
    WW       = np.array([2400.0,2100,1800,1500,1200,1000],dtype=int)
    LO_predt = hyp_inv(p_best,WWT)
    LO_pred  = hyp_inv(p_best,WW) 
    return WW,LO_pred,LO_predt


# -----------------------------------------------------------
# Simple least-squares estimate
# Same as using numpy.linalg.lstsq (https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
# -----------------------------------------------------------
def LS(X,Si,Y):
    A    = np.dot(np.dot(X.T,Si),X)
    b    = np.dot(np.dot(X.T,Si),Y)
    Ainv = np.linalg.inv(A)
    par  = np.dot(Ainv,b)
    return par,np.sqrt(np.diag(Ainv))

# -----------------------------------------------------------
# Iterative polynomial fit with outlier removal
# Documentation: https://docs.astropy.org/en/stable/api/astropy.modeling.fitting.FittingWithOutlierRemoval.html
# -----------------------------------------------------------
def poly_fit(x,y,deg,sig,n_iter=3):

    """
    --> Inputs:     - x:      X vector
                    - y:      Data to fit
                    - deg:    Degree of the polynomial to fit
                    - sig:    Threshold for the outlier removal [in sigma]
                    - n_iter: Number of iterations for the fit

    --> Outputs:    - fitted_model (function): best-fitting model after last iteration
                    - filtered_data: data after outlier removal
    """
    
    pol_f     = polynomial.Polynomial1D(deg) ### Init polynom
    fit       = fitting.LinearLSQFitter()   ### Init optim method
    or_fit    = fitting.FittingWithOutlierRemoval(fit,sigma_clip,niter=n_iter, sigma=sig)  ### Do the fit at sig sigma level
    or_fitted_model,mask = or_fit(pol_f,x,y) 
    filtered_data        = np.ma.masked_array(y,mask=mask)
    fitted_model         = fit(pol_f,x,filtered_data)
    return fitted_model,filtered_data






# -----------------------------------------------------------
# Apply PCA to an input sequence of spectra:
# 1. Apply PCA to the sequence (centered + reduced)
# 2. Set the first N_comp_pca components to 0 (i.e. components associated to largest variance contribution in the data)
# 3. Project back to the input 'data frame'
# -----------------------------------------------------------
def make_pca(I,N_comp_pca,return_all=False):

    """
    --> Inputs:     - I:          Input sequence of spectra (2D matrix)
                    - N_comp_pca: Number of PCA components to remove
                    - return_all: Boolean --> if True, return discared components back into the data frame

    --> Outputs:    - e_var: Relative contribution of each component to the variance in the data
                    - I_pca: Sequence of spectra after removing the first N_comp_pca
                    - I_del: Removed components (same shape as I) -- NOTE: only if return_all == True
                    
    """

    ### Preprocessing: center + reduce input matrix
    I2       = np.copy(I)
    Im       = I2.mean(axis = 0)
    Is       = I2.std(axis = 0)

    ### PCA decomposition
    U, s, VT = np.linalg.svd((I2-Im)/Is)
    comp     = VT[:len(I2)]
    comp2    = np.copy(comp)
    var      = s ** 2 / I2.shape[0]
    e_var    = var/var.sum()
    X_new    = s * U

    ### Set the first components to 0
    comp2[:N_comp_pca]  = np.zeros((N_comp_pca,len(comp[0])))

    ### Project back into initial basis
    I_pca               = Is*np.dot(X_new,comp2) + Im

    if return_all: # compute individually each removed components
        I_del  = np.zeros((N_comp_pca,len(I),len(I[0])))
        for ii in range(N_comp_pca):
            cc        = np.zeros(I.shape)   
            cc[ii]    = comp[ii]
            I_del[ii] = np.dot(X_new,cc)
        return e_var,I_pca,I_del
    else: return e_var,I_pca




