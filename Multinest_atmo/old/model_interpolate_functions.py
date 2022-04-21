import numpy as np
from numpy import ma

import sys
import os
import time

from scipy import interpolate
import batman

#import matplotlib.pyplot as plt
#import matplotlib



class Model_2D:
    
    """
    Class model
    Store the templates of transmission spectrum of the planet atmosphere
    """
    
    
    def __init__(self,reduced_dic):
        
        
        self.nord = reduced_dic["nord"]
        self.Wm = reduced_dic["wavelength"]
        self.absor = reduced_dic["absorption"]
        self.Fm = []

    def create_interp(self):
        """
        Create a 2D sequence template
        Interpolate each template         
        """       
        self.Fm = interpolate.interp1d(self.Wm,self.absor,kind='linear')

    
#    def build_model(self,window):
#        """
#        Create a 2D sequence template
#        Interpolate each template         
#        """
#        
#        I_t   = self.absor
#        I_mod = []
#        for nn in range(len(window)):
#            I_line = I_t*window[nn]
#            I_mod.append(I_line)
#        I_mod = np.array(I_mod,dtype=float)
#                
#        # Compute interpolation of each line of the model
#        FM = []
#        for n in range(len(window)):
#            f_mod = interpolate.interp1d(self.Vm,I_mod[n],kind='linear')
#            FM.append(f_mod)
#        self.Fm = np.array(FM)



        
class Planet:

    """
    Class Planet
    All elements related to the observed transit and the weighting window to make the model
    """
    
        
    def __init__(self,Rp_Rs,config,dates):
        
        
        self.rp    = Rp_Rs  ## Planet radius in stellar radius unit [R_s]
        self.inc   = config["inc"] ## Inclination of the planetary transit [deg]
        self.t0    = config["t0"]  ## Mid-transit time [phase]
        self.a     = config["sma"]  ## SMA [R_s]
        self.per   = config["orb_per"] ## Orbital period [d]
        self.ecc   = config["ecc"]  ## eccentricity
        self.w     = config["w_peri"]  ## long. of periastrion [deg]
        self.ld    = config["limbdark"]   ## limb darkening model -- batman
        self.u     = config["u_limbdark"]   ## factors for limb darkening
        
        self.date   = np.array(dates)   ## Phase vector
        self.batman = ""   ## Batman transit model
        self.flux   = []   ## Normalized light curve
        self.window = []   ## Window function from transit curve
        self.lims   =  []   ## limits where the window is non zero
        
        
    def make_batman(self):
        """
        Use batman to generate a transit light-curve given the attributes of the Planet object
        """        
        
        ### See https://www.cfa.harvard.edu/~lkreidberg/batman/
        ### Init parameters
        params = batman.TransitParams()
        params.rp        = self.rp                       
        params.inc       = self.inc
        params.t0        = self.t0   
        params.a         = self.a    
        params.per       = self.per  
        params.ecc       = self.ecc 
        params.w         = self.w          
        params.limb_dark = self.ld
        params.u         = self.u

        
        ### Make transit light-curve
        self.batman = batman.TransitModel(params,self.date)
        self.flux   = self.batman.light_curve(params)
  
    def make_window(self):
        """
        Build weighting window from the transit-light curve
        """        

        ### Build window
        FF          = 1.-self.flux
        self.window = FF/np.max(FF)
        lims = np.where(self.window>0.2)[0]
        self.start_transit = lims[0]
        self.end_transit = lims[-1]
            
            
    def RVP(self,kp,v0):
        """
        Return RV values for the planet
        """
        V1 = kp*np.sin(2*np.pi*(self.date-self.t0)/(self.per))+v0
        return V1



class total_model:
    
    """
    Class CCF that handles the Matching template filter and the cross-correlation between the modelled sequence of spectra and the data
    """

    
    def __init__(self,Kp,Vsys,orders,Wmean,models,V,Planet,ddv):
        
        
        self.planet = Planet   ## Planet object
        self.orders = orders
        self.Wmean = Wmean
        self.models  = models   ## Model object
        
        self.V = V

        
        self.Kp  = Kp  ## Values of semi-amplitude of the planet orbit for parameter search (1D vector)
        self.Vsys = Vsys   ## Values of radial velocity at mid-transit for parameter search (1D vector)
        self.ddv    = ddv  ## Vector of velocity used for the integration of the model when binned into the data sampling scheme
                           ## We generally center it on 0 with half width of 1 SPIRou pixel (i.e. ~2 km/s)  
        
        self.corrcoeff = 0
        
    def fill_model(self):
        """
        Only keep the models that are needed for a given transit
        Could be seriously improved
        """
        
        true_mod = []
        for i in range(len(self.orders)):
            #print(i)
            for j in range(len(self.models)):
                #print(j)
                if self.orders[i] == self.models[j].nord:
                    true_mod.append(self.models[j])
        self.models = true_mod      

    def bin_model(self):

        """
        Bin the model at the resolution of the data accounting for the shifts in velocity for (k,v) values
        
        Inputs:
        - k,v:    semi-amplitude and mid-transit velocity (floats)
        - V_data: Velocity matrix of the sequence to bin (returned after applying OBS.shift_rv method.
                  Each line is the velocity vector of the spectrum shifted in the stellar rest frame

        Outputs:
        - I_ret: Binned model at the resolution of the data where spectra are shifted at (k,v)
        """

        ### Compute the Radial Velocity of the planet in the stellar rest frame
        DVP   = self.planet.RVP(self.Kp,self.Vsys)
         

        num_spectra = len(self.planet.date)
        
        data_ret = []
        model_ret = []  ### Init binned sequence of spectra
        std_ret  = []
        c0      = 29979245800.0e-5
        #for i in range(len(self.orders)):
            #print(self.models[i].nord)


        for i in range(len(self.orders)):
            V_data =  self.V[i]
            for n in range(num_spectra):

                if (self.planet.window[n] > 0.2):
                    I_tmp= np.zeros(len(V_data))
                    data_tmp = np.zeros(len(V_data))
    
    			### For dd in the window centered on 0 and of 1 px width (here 2 km/s)
                    # print(self.models[i].nord,self.Wmean[i])
                    for dd in self.ddv:
                        #print(((V_data+dd-DVP[n])/c0+1.0)*self.Wmean[i])
                        #print(np.min(((V_data+dd-DVP[n])/c0+1.0)*self.Wmean[i]),np.max(((V_data+dd-DVP[n])/c0+1.0)*self.Wmean[i]))
                        I_tmp += self.models[i].Fm(((V_data+dd-DVP[n])/c0+1.0)*self.Wmean[i])*self.planet.window[n] 
                    I_tmp = I_tmp/len(self.ddv)### Average values to be closer to measured values
                    I_tmp -= np.mean(I_tmp)
                    model_ret.append(I_tmp)
                    
        # np.savetxt("lol.txt",I_ret)
        
        
        
        return model_ret ### Binned modelled sequence shifted at (kp,v0)


        # 
        # V_mod   = c0*(self.Wm/self.W_mean-1)
        # self.Vm = V_mod 

    # def make_corr(self,V_data,I_data):

    #     """ 
    #     Explore (Kp,V0) parameter space by correlating the modelled sequence of spectra with the reduced sequence
    #     for all couple of parameters in the grid. 

    #     Inputs:
    #     - V_data: Velocity matrix where each line is the velocity vector of the spectrum shifted in the stellar rest frame
    #     - I_data: reduced sequence of spectra

    #     Outputs:
    #     - corr: Matrix of correlation coefficients (shape: (len(self.K_vec),len(self.V0_vec)))
    #     """

    #     Im = self.bin_model(self.Kp,self.Vsys,V_data)
    #     corr = np.array(get_cc(I_data,Im),dtype=float)

        
    #     self.model2D = Im
    #     self.corrcoeff = corr


# def get_cc(Yd,Ym):
#     """
#     Compute the correlation coefficient between the sequence of spectra and the modelled sequence of the spectra
#     If a spectrum in the modelled sequence of spectra is 0, np.corrcoef returns NaN. This displays a warning message,
#     but we account for this in the process.
#     Inputs:
#     - Yd: 2D sequence of spectra
#     - Ym: Modelled sequence of spectra (binned at the resolution of data - same shape as Yd)

#     Outputs:
#     - Correlation coefficient between the 2 spectra
#     """

#     C0 = np.zeros(len(Yd))
#     for n in range(len(Yd)):
#         #c = np.ma.corrcoef(Yd[n],Ym[n]).data[0,1]
#         c = np.corrcoef(Yd[n],Ym[n])[0,1]
#         if np.isfinite(c): C0 [n]= c  ### Avoid NaNs (modelled spectrum is 0 (no planet in out-of-transit periods)
#     return C0











        
