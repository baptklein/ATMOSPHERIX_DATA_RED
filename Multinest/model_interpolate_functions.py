import numpy as np
from numpy import ma

import sys
import os
import time

from scipy import interpolate

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


class total_model:
    
    """
    Class CCF that handles the Matching template filter and the cross-correlation between the modelled sequence of spectra and the data
    """

    
    def __init__(self,Kp,Vsys,models,orders,Wmean,V,phase,window,Vstar,ddv,proj):
        
        
        self.orders = orders
        self.Wmean = Wmean
        self.models  = models   ## Model object
        self.phase = phase
        self.window = window
        
        self.V = V
        self.Vstar = Vstar

        
        self.proj = proj
        #print(len(orders),len(self.proj))
        #print(np.shape(self.proj[0]))       
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
        DVP   = self.Kp*np.sin(2.0*np.pi*self.phase)+self.Vsys+self.Vstar
         

        num_spectra = len(self.window)
        
        data_ret = []
        model_tot = []  ### Init binned sequence of spectra
        std_ret  = []
        c0      = 29979245800.0e-5
        #for i in range(len(self.orders)):
            #print(self.models[i].nord)

        
        for i in range(len(self.orders)):
#            print(self.orders[i],np.shape(self.proj[i]))
            mask = []
            V_data =  self.V[i]
            #total shape of the model
            model_ret = np.zeros((num_spectra,len(V_data)))
            for n in range(num_spectra):
                if (self.window[n] > 0.2):
                    mask.append(n)
                    I_tmp= np.zeros(len(V_data))
                    data_tmp = np.zeros(len(V_data))
    
    			### For dd in the window centered on 0 and of 1 px width (here 2 km/s)
                    # print(self.models[i].nord,self.Wmean[i])
                    for dd in self.ddv:
                        #print(((V_data+dd-DVP[n])/c0+1.0)*self.Wmean[i])
                        #print(np.min(((V_data+dd-DVP[n])/c0+1.0)*self.Wmean[i]),np.max(((V_data+dd-DVP[n])/c0+1.0)*self.Wmean[i]))
                        I_tmp += self.models[i].Fm(((V_data+dd-DVP[n])/c0+1.0)*self.Wmean[i])*self.window[n] 
                    I_tmp = I_tmp/len(self.ddv)### Average values to be closer to measured values
                    #I_tmp = np.mean(I_tmp)
                    model_ret[n] = I_tmp
                else:
                    model_ret[n] = np.zeros(len(V_data))
            try:
                #mean_to_keep = np.mean(model_ret,axis=1)
                #model_ret = (model_ret.T-mean_to_keep).T
                Il    = np.log(model_ret+1.0)
                im    = 0.0
                ist   = 1.0 
#                im    = np.tile(np.nanmean(Il,axis=0),(len(Il),1))
#                ist   = np.tile(np.nanstd(Il,axis=0),(len(Il),1))

                ff    = (Il - im)/ist
                model_ret= np.exp((ff-np.matmul(self.proj[i],ff))*ist+im)-1.0
                model_ret = model_ret.T
                model_ret = model_ret[50:-50]
                model_ret -= np.mean(model_ret,axis=0)
                model_ret = model_ret.T
            except:
                np.save("/home/fdebras/tmp/proj.npy",self.proj[i])
                np.save("/home/fdebras/tmp/model_ret.npy",model_ret)
                sys.exit()
                print("oups")
                pass
            mask = np.array(mask)
            model_tot = model_tot+(model_ret[mask].tolist())
        # np.savetxt("lol.txt",I_ret)
        return model_tot ### Binned modelled sequence shifted at (kp,v0)


