import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from astropy.modeling import fitting, polynomial
from astropy.stats import sigma_clip
import pickle

from sklearn.decomposition import PCA
from wpca import WPCA

c0 = 299792.458 #km/s


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
        I_cl    = np.delete(self.I_raw,ind_tel,axis=1)
        A_cl    = np.delete(self.I_atm,ind_tel,axis=1)
        W_cl    = np.delete(self.W_raw,ind_tel)
        return W_cl,I_cl,A_cl




    
    
        
    # -----------------------------------------------------------
    # This function corrects for Rossiter-Maclaughlin and center-
    # to-limb variations in the stellar line profiles. For the 
    # relevant wavelength ranges, the function takes a synthetic
    # spectrum in which the relevant planet-induced perturbations
    # have been corrected, and correct for these effects by normali-
    # sing the observed spectra by the synthetic user-provided spec.
    # -----------------------------------------------------------
    def correct_star(self,VS_corr,IS_corr,V_shift,sig_g=2.28):
        """
        --> Inputs:     - V_obs:     Velocity vector of the observed spectra [km/s]
                        - I_obs:     Observed spectra (2D matrix)
                        - VS_corr:   Velocity vector of the synthetic spectra (1D)
                        - IS_corr:   Synthetic spectra used to correct the observed ones (2D matrix)
                        - V_shift:   Geocentric to barycentric RV correction [km/s]
                        - I_atm:     Normalised telluric spectrum
                        - sig_g:     STD of SPIRou instrumental profile [km/s]
    
    
        --> Outputs:    - If:        Sequence of spectra corrected from IS_corr
        """
        
        #V_obs,I_obs,I_atm = self.V_cl,self.I_cl,self.A_cl
        V_obs,I_obs = self.V_cl,self.I_cl
        I_obs = (I_obs.T/np.max(I_obs,axis=1)).T
        dddv    = np.linspace(-3000.*sig_g,3000.*sig_g,30)
        G       = normal_law(dddv,0.0,sig_g*1000.)
        step    = dddv[1]-dddv[0]
        If      = np.copy(I_obs)
        for uu in range(len(I_obs)):
        
            ### Select only relevant parts for the correction
            iss   = np.where((V_obs>=VS_corr.min())&(V_obs<=VS_corr.max()))[0]
            vv    = V_obs[iss]
            ii    = I_obs[uu,iss]
            # aa    = I_atm[uu,iss]    
    
            ### Select only the lines with negligible telluric absorption
            # it    = np.where(aa>1-thres_tel)[0]
            V_sel = vv#[it]
            I_sel = ii#[it]        
    
            ### Shift synthetic spectra in Geocentric frame
            fi       = interp1d(VS_corr,IS_corr[uu],kind="linear",fill_value="extrapolate")
            Icg      = step * (fi(V_sel-V_shift[uu]+dddv[0])*G[0]+fi(V_sel-V_shift[uu]+dddv[-1])*G[-1]) * 0.5
            for hh in range(1,len(dddv)-1):
                Icg += step*fi(V_sel-V_shift[uu]+dddv[hh])*G[hh]  
            
            ### Linear fit of the model to the data
            idd        = np.where(Icg<np.median(Icg)-0.5*np.std(Icg))  ## Fit the lines only
            X          = np.array([np.ones(len(V_sel))[idd],Icg[idd]],dtype=float).T
            p,pe       = LS(X,I_sel[idd])
            
            
            ### Shift full synthetic spectra and correct for it
            Icf      = step * (fi(vv-V_shift[uu]+dddv[0])*G[0]+fi(vv-V_shift[uu]+dddv[-1])*G[-1]) * 0.5
            for hh in range(1,len(dddv)-1):
                Icf += step*fi(vv-V_shift[uu]+dddv[hh])*G[hh]  
            Xc          = np.array([np.ones(len(vv)),Icf],dtype=float).T
            Ip          = np.dot(Xc,p) 
            If[uu,iss] /= Ip        
            
        return If
    


    def master_out(self,V_corr,n_ini,n_end,sig_g,N_bor):      
        ### If the order is kept - Remove high-SNR out-of-transit reference spectrum    
        ### Start by computing mean spectrum in the stellar rest frame
        I_bary    = move_spec(self.V_cl,self.I_cl,V_corr,sig_g)  ## Shift to stellar rest frame      
        I_med     = np.median(np.concatenate((I_bary[:n_ini],I_bary[n_end:]),axis=0),axis=0) ## Compute median out-of-transit   
        I_med_geo = move_spec(self.V_cl,np.array([I_med]),-1.*V_corr,sig_g)  ## Move back ref spectrum to Geocentric frame
        I_sub1    = np.zeros(self.I_cl.shape)
        for kk in range(len(self.I_cl)):
            X          = np.array([np.ones(len(I_med_geo[kk])),I_med_geo[kk]],dtype=float).T
            p,pe       = LS(X,self.I_cl[kk])
            Ip         = np.dot(X,p)
            I_sub1[kk] = self.I_cl[kk]/Ip
            
        ### Then compute reference spectrum in the Geocentric frame
        I_med2  = np.median(np.concatenate((I_sub1[:n_ini],I_sub1[n_end:]),axis=0),axis=0) 
        I_sub2  = np.zeros(I_sub1.shape)
        for kk in range(len(I_sub1)):
            X          = np.array([np.ones(len(I_med2)),I_med2],dtype=float).T
            p,pe       = LS(X,I_sub1[kk])
            Ip         = np.dot(X,p)
            I_sub2[kk] = I_sub1[kk]/Ip    
            
            
        ### Remove extremities to avoid interpolation errors
        W_sub = self.W_cl[N_bor:-N_bor]
        I_sub = I_sub2[:,N_bor:-N_bor]
        
        return W_sub, I_sub
     


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
    def normalize(self,Ws,Is,N_med,sig_out,N_adj,nbor):

        """
        --> Inputs:     - Order object
                        - Ws:      Wavelength vector
                        - Is:      2D median-normalised flux matrix
                        - N_med:   Cut-off frequency of the median filter (nb of points of the sliding window of
                                   the moving median)
                        - sig_out: Threshold for outlier removal (in sigma)
                        - N_adj:   Number of adjacent points to each outlier removed with the outlier 
                        - nbor:    Number of points removed at each edge of the order

        --> Outputs:    - self.I_norm, self. W_norm
        """

        ### Initialization
        ind_out_fin = []
        I_norm      = np.zeros((len(Is),len(Ws)-int(2*nbor)))
        I_corr      = np.zeros(Is.shape)
        I_med_tot   = np.zeros(Is.shape)
        
        for ii in range(len(Is)):
            I           = np.copy(Is[ii]) # Temp. spectrum
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
            I_norm[ii] = In[nbor:-nbor] # Remove pts at both edge of the order
        W_norm = Ws[nbor:-nbor]  
        return W_norm,I_norm





    # -----------------------------------------------------------
    # Detrending with airmass
    # -- Step 1.5 of the data reduction procedure
    #  Fit a polynomial model of airmass to the sequence of normalized
    #  spectra
    # -----------------------------------------------------------
    def detrend_airmass(self,W,I,airmass,deg=2):

        """
        --> Inputs:     - Order object
                        - airmass: vector of airmass
                        - deg:     Degree of the airmass model (2 in generally sufficient)


        --> Outputs:    - self.I_det
                        - pb: vector of best-fitting spectra
                        - I_pred: Best-fitting modelled sequence of spectra
        """

        indw    = np.argmin(np.abs(W-self.W_mean))
        COV_inv = np.diag(1./np.std(I[:,indw-200:indw+200],axis=1)**(2)) ## Assumes that normalized spectra dominated by white noise
        X       = np.ones((len(I),deg+1))
        for ii in range(deg): X[:,ii+1] = airmass**(ii+1)
        pb,pbe = LS(X,I,COV_inv)
        I_pred = np.dot(X,pb)
        I_det  = I-I_pred
        return I_det




    # -----------------------------------------------------------
    # Filter bad pixels
    # -- Step 1.6 of the data reduction procedure
    #  1. Compute standard deviation for each pixel
    #  2. Fit a parabola to the distribution of pixel dispersion
    #  3. Remove all outliers above fit
    # -----------------------------------------------------------
    def filter_pixel(self,W,I,deg_px=2,sig_px=4.):

        """
        --> Inputs:     - Order object
                        - W:       Wavelength vector
                        - I:       Intensity values
                        - deg_px:  degree of the polynomial fit (2 is generally sufficient)
                        - sig_px:  Threshold for outlier removal

        --> Outputs:    - self.I_red, self.W_red
        """

        n_iter_fit    = 10 ### Number of iterations for the iterative polynomial fit with outliers removal
                           ### See documentation on astropy --> https://docs.astropy.org/en/stable/api/astropy.modeling.fitting.FittingWithOutlierRemoval.html
        rms_px        = I.std(axis=0) # Dispersion of each pixel (computed along the time axis)
        WW            = W - np.mean(W) # Wavelength vector for the fit
        model,filt_px = poly_fit(WW,rms_px,deg_px,sig_px,n_iter_fit) ### See functions below
        rms_pred      = model(WW) # Best prediction

        ### Identify and remove outliers
        ind_px        = []
        for ii in range(len(filt_px)):
            if filt_px[ii] != "--": ind_px.append(ii)
            elif rms_px[ii] < rms_pred[ii]: ind_px.append(ii)
        return W[ind_px],I[:,ind_px],ind_px



    # -----------------------------------------------------------
    # Automatically tune the number of PCA components removed:
    # Generate sequence of spectra containing only white noise
    # amplified by the blaze function. Apply PCA to Nmap white 
    # noise maps and store the highest eigenvalue. The components
    # to remove are those whose eigenvalue is larger than a threshold
    # computed from the largest eigenvalues of the noise maps.
    # -----------------------------------------------------------

    def tune_pca(self,mode,factor,Nmap=5,min_pca=0):

        """
        --> Inputs:     - Order object
                        - mode: way to remove mean and std. "none" means no removal, "total" means 
                        removal of global mean and std, "per_pix" means removal of std and mean by column (per pixel)
                        and "per_obs" by line (per observation)
                        - factor: multiplicative factor to compare the eigenvalues with white noise eigenvalues
                                  Each eigenvalues above factor*white_eigenvalue is suppresed
                        - Nmap:    Number of white noise map used to compute the threshold 

        --> Outputs:    - ncf:     Number of PCA components to remove
        """    

        N_px          = 200    ### Half nb of px used to compute the dispersion for each pixel
        n_iter_fit    = 10     ### Number of iterations for the polynomial fit to the px std
        
        ### Initialisation:
        Il            = np.log(self.I_fin)
        
        if mode =="none":
            ff = Il
            im = 0.0
            ist = 1.0
        elif mode =="global":
            im            = np.nanmean(Il)
            ist           = np.nanstd(Il)   
            ff            = (Il - im)/ist     
            
        elif mode == "per_pix":

            im = np.tile(np.nanmean(Il,axis=0),(Il.shape[0],1))
            ist = np.tile(np.nanstd(Il,axis=0),(Il.shape[0],1))
            
            ff = (Il-im)/ist
        
        elif mode == "per_obs":

            im = np.tile(np.nanmean(Il,axis=1),(Il.shape[1],1)).T
            ist = np.tile(np.nanstd(Il,axis=1),(Il.shape[1],1)).T
            
            ff = (Il-im)/ist
  
   
        
        ### Determinate S/N at the center of the order for each epoch
        indw          = np.argmin(np.abs(self.W_fin-self.W_fin.mean())) 
        std_mes       = np.std(ff[:,indw-N_px:indw+N_px],axis=1)
        
        ### Determine the blaze amplification function (border of the order)
        WW            = self.W_fin - self.W_mean
        std_px        = np.std(ff,axis=0)
        std_in        = np.dot(std_mes.reshape((len(ff),1)),np.ones((1,len(self.W_fin))))
        model,filt    = poly_fit(WW,std_px,2,5,n_iter_fit)
        ampl          = model(WW)/np.min(model(WW))
        
        ### Generate noise maps, amplify them, and apply PCA
        thres         = np.zeros(Nmap) ### Store highest eigenvalue for each noise map
        for ii in range(Nmap):        
            ### Generate noise map
            NN    = np.random.normal(0.0,std_in*ampl)
            Nm    = np.dot(np.mean(NN,axis=0).reshape((NN.shape[1],1)),np.ones((1,NN.shape[0]))).T
            Ns    = np.dot(np.std(NN,axis=0).reshape((NN.shape[1],1)),np.ones((1,NN.shape[0]))).T  
            Nf    = (NN-Nm)/Ns          
            
            ### Apply PCA
            pca   = PCA(n_components=len(Nf))
            pca.fit(np.float32(Nf))
            var       = pca.explained_variance_ratio_    
            
            ###Store highest eigenvalue     
            thres[ii] = np.max(var)
    
        ### Apply PCA to observed data
        pca   = PCA(n_components=len(NN))
        x_pca = np.float32(ff)
        pca.fit(x_pca)       
        var   = pca.explained_variance_ratio_ 

        ### Nb of components: larger than factor*max highenvalue
        ncf   = max(min_pca,len(np.where(var>factor*np.max(thres))[0]))

        return ncf
        



def read_data_and_create_list(filename):
    print("Read data from",filename)
    with open(filename,'rb') as specfile:
        orders,WW,Ir,blaze,Ia,T_obs,phase,window,berv,vstar,airmass,SN = pickle.load(specfile)
    nord     = len(orders)
    print(nord,"orders detected")
    list_ord = []
    for nn in range(nord):
        O        = Order(orders[nn])
        O.W_raw  = np.array(WW[nn],dtype=float)
        O.I_raw  = np.array(Ir[nn],dtype=float)
        O.blaze  = np.array(blaze[nn],dtype=float)    
        O.I_atm  = np.array(Ia[nn],dtype=float)
        O.SNR    = np.array(SN[nn],dtype=float)
        O.W_mean = O.W_raw.mean()
        list_ord.append(O)
    print("DONE\n")
    return T_obs,phase,window,berv,vstar,airmass,SN,list_ord

def get_transit_dates(wind):

    """
    --> Inputs:     - flux:  Relative transit flux (e.g., output from batman python module)

    --> Outputs:    - n_ini: Index of the last point before transit
                    - n_end: Index of the first point after transit
    """

    n_ini,n_end = 1,1
    if wind[0] > 0.0: n_ini = 0
    else: 
        cc = 0
        while wind[cc] == 0.0:
            cc += 1
        n_ini = cc-1
    if wind[-1] > 0.0: n_end = len(wind)-1
    else:
        cc = n_ini + 1
        while wind[cc] > 0.0:
            cc += 1
        n_end = cc
    return n_ini,n_end


    
def tellurics_and_borders(O,dep_min,thres_up,N_bor): 
    ### First we identify strong telluric lines and remove the data within these lines -- see Boucher+2021
    W_cl,I_cl,A_cl =  O.remove_tellurics(dep_min,thres_up)
    
    W_cl = W_cl[N_bor:-N_bor]
    I_cl = I_cl[:,N_bor:-N_bor]
    A_cl = A_cl[:,N_bor:-N_bor]
    # W_cl,I_cl = O.filter_pixel(W_cl,I_cl,deg_px,sig_out)
        
    ### If the order does not contain enough points, it is discarded

    return W_cl,I_cl,A_cl
        

# -----------------------------------------------------------
# Get transit initial and end dates from an input transit curve
# -----------------------------------------------------------


#outdated
# def stellar_from_file(O,nn,IC_name,WC_name,V_brog,sig_g=2.28):        
       
#     ### Correction of stellar contamination (RM and center-to-limb variations)
#     IS_corr = np.load(IC_name)
#     WS_corr = np.load(WC_name)
#     VS_corr = c0*(WS_corr/O.W_mean-1.)  
    
#     if (WS_corr.min()>=O.W_cl.max()) or (WS_corr.max()<=O.W_cl.min()):
#         print("No stellar correction available for order:",nn) 
#         return O.I_cl
#     else:
#         cov = 100.*(WS_corr.max() - WS_corr.min())/(O.W_cl.max()-O.W_cl.min())
#         print("STELLAR CORRECTION: [",WS_corr.min(),",",WS_corr.max(),"]")
#         print("Order wavelengths: [",O.W_cl.min(),",",O.W_cl.max(),"] - Coverage:",round(cov,1),"%")
#         return O.correct_star(VS_corr,IS_corr,V_brog,sig_g=sig_g)

        

# -----------------------------------------------------------
# Move spectra from one frame to another
# -----------------------------------------------------------    
def move_spec(V,I,Vc,sig_g):
    """
    --> Inputs:     - V:     Velocity vector (assumed 1D)
                    - I:     Array of flux values (assumed 2D [N_obs,N_wav])
                    - Vc:    Velocimetry correction [km/s]
                    - pixel: Binned instrument pixel in wavelength space
                    - kind:  kind of interpolatation (scipy interp1D)  
    

    --> Outputs:    - I_al:  2D matrix of Vc-corrected spectra
    
    """

    I_al    = np.zeros((len(Vc),len(V)))
    dddv    = np.linspace(-3.*sig_g,3.*sig_g,30)
    G       = normal_law(dddv,0.0,sig_g)
    step    = dddv[1]-dddv[0]
    for ii in range(len(Vc)):
        ### Depending on which frame we're moving into
        if len(I) == len(Vc): fi = interp1d(V,I[ii],kind="cubic",fill_value="extrapolate")
        else:                 fi = interp1d(V,I[0],kind="cubic",fill_value="extrapolate")
        I_tmp     = step * (fi(V+Vc[ii]+dddv[0])*G[0]+fi(V+Vc[ii]+dddv[-1])*G[-1]) * 0.5
        for hh in range(1,len(dddv)-1):
            I_tmp += step*fi(V+Vc[ii]+dddv[hh])*G[hh]  
        I_al[ii] = I_tmp          
    return I_al 


#outdated
# def airmass_correction(O,airmass,deg_airmass):        
#     I_log           = np.log(O.I_norm2)
#     I_det_log       = O.detrend_airmass(O.W_norm2,I_log,airmass,deg_airmass)
#     I_det           = np.exp(I_det_log)
#     return I_det




def apply_PCA(O,mode_norm_pca,wpca):
    if O.n_com > 0:
        Il    = np.log(O.I_fin)
        
        if mode_norm_pca =="none":
            ff = Il            
            im = np.zeros(ff.shape)
            ist = np.ones(ff.shape)
            
        elif mode_norm_pca =="global":
            im            = np.nanmean(Il)
            ist           = np.nanstd(Il)   
            ff            = (Il - im)/ist     
            
        elif mode_norm_pca == "per_pix":

            im = np.tile(np.nanmean(Il,axis=0),(Il.shape[0],1))
            ist = np.tile(np.nanstd(Il,axis=0),(Il.shape[0],1))            
            ff = (Il-im)/ist
        
        elif mode_norm_pca == "per_obs":

            im = np.tile(np.nanmean(Il,axis=1),(Il.shape[1],1)).T
            ist = np.tile(np.nanstd(Il,axis=1),(Il.shape[1],1)).T            
            ff = (Il-im)/ist
        
        #weighted PCA with the weight being the variance of the pixels
        if wpca:
            weight =1./np.std(ff,axis=0)
            weight = weight*np.ones_like(ff)
            pca   = WPCA(n_components=O.n_com)
            x_pca = np.float32(ff)
            pca.fit(x_pca,weights=weight)
            
        #else normal pca
        else:
            pca   = PCA(n_components=O.n_com)
            x_pca = np.float32(ff)
            pca.fit(x_pca)                    
        
        
        principalComponents = pca.transform(x_pca)
        x_pca_projected = pca.inverse_transform(principalComponents)        
        I_pca = np.exp((ff-x_pca_projected)*ist+im)-1.0
        # I_pca = np.exp((ff-x_pca_projected))-1.0

        print(O.n_com,"PCA components discarded")
        
        #For Gibson2021
        try:
            inver = np.linalg.pinv(principalComponents)
            proj = np.matmul(principalComponents,inver)
        except:
            proj = np.zeros((len(O.I_fin),len(O.I_fin)))
    else:
        print("0 PCA components discarded")
        I_pca = O.I_fin
        proj  = np.zeros((len(O.I_fin),len(O.I_fin)))

    return I_pca,proj
                
        
def normal_law(v,mu,sigma):
    g = 1./(np.sqrt(2.*np.pi)*sigma) * np.exp(-0.5*((v-mu)/(sigma))**(2))
    return g
    



# -----------------------------------------------------------
# Simple least-squares estimate
# Same as using numpy.linalg.lstsq (https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
# -----------------------------------------------------------
def LS(X,Y,Si=[]):

    if len(Si) > 0:
        A    = np.dot(np.dot(X.T,Si),X)
        b    = np.dot(np.dot(X.T,Si),Y)
    else:
        A    = np.dot(X.T,X)
        b    = np.dot(X.T,Y)    
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


def calculate_final_metrics(O,N_px,file):
    ### ESTIMATES FINAL METRICS
    indw          = np.argmin(np.abs(O.W_fin-O.W_fin.mean())) 
    O.SNR_mes     = 1./np.std(O.I_fin[:,indw-N_px:indw+N_px],axis=1) 
    O.SNR_mes_pca = 1./np.std(O.I_pca[:,indw-N_px:indw+N_px],axis=1)        
    
    txt = str(O.number) + "  " + str(len(O.W_fin)) + "  " + str(np.mean(O.SNR)) + "  " + str(np.mean(O.SNR_mes)) + "  " + str(np.mean(O.SNR_mes_pca)) + "  " + str(O.n_com) + "\n"
    file.write(txt)            
       
        

