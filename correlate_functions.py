#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:39:45 2023

@author: florian
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.signal
from scipy import stats
import pickle
import time
import speed_functions as speed


from global_parameters import c0

def load_data(filename,select_ord,list_ord1,dire_mod):
    """
    Load data from a pkl reduced data file. 

    Parameters:
    filename (str): The name of the data file to be loaded.
    select_ord (list, optional): List of selected orders. If not provided, all orders are used.
    
    Returns:
    tuple: A tuple containing various data arrays and lists extracted from the file:
               list_ord: the list of orders used
               wl : the list of wavlenght, size len(list_ord)
               data_tot: the list of 2D (time*wavelength) intensity file, size len(list_ord)
               phase: numpy array of the phases of transit
               window: numpy array of the transit window
               Stdtot: the list of 1D (wavelength) std of data_tot along the time axis, size len(list_ord)
               SNRtot: the list of 1D (time) averaged SNR per order, size len(list_ord) 
               projtot: the list of the PCA projectors from Gibson et al. 2022, size len(list_ord)
               F: the list of the interpolated models order by order, size len(list_ord)
               Vc: numpy array of stellar speed, size len(list_ord)
               berv: numpy array of BERV, size len(list_ord)

           
    """
    with open(filename,'rb') as ccfile:
        orders,W_data,I_data,T_obs,phase,window,berv,Vc,SN,proj = pickle.load(ccfile)
    list_ord = []
    
    #The data directory contains the data after reduction by  Baptiste's code
    #THe mod directory contains the templates in ntwo columns : wl and (1-rp/Rs**2) normalised
    wl = []
    data_tot=[]
    F =[]
    Stdtot = []
    SNRtot = []
    projtot = []


    k  = 0
    for no in orders:
        if select_ord:
            if no in list_ord1:
                list_ord.append(no)
                pass
            else:
                k=k+1
                continue
        else:
            list_ord.append(no)
    
        file_mod = dire_mod+"/template"+str(no)+".txt"
    
        try:
            mod = np.loadtxt(file_mod)
        except:
            print("Careful ! Headers on model. Assuming 5 lines")
            mod = np.loadtxt(file_mod,skiprows=5)
            
        # mod = mod[::3]
        Std = np.zeros(np.shape(I_data[k])[1])
        for i in range(np.shape(I_data[k])[1]):
            Std[i] = (np.std(I_data[k][:,i]))

        
        # Stdfit = np.poly1d(fit2)
        data_tot.append(I_data[k])
        wl.append(W_data[k])
        
        #Decode whether tou use the actual Std or the polynomial fit 
        #of the Std (see footnote in the paper)
        Stdtot.append(Std)
        SNRtot.append(SN[k])
        projtot.append(proj[k])
    
        #prepare model interpolation
        f = interp.interp1d(mod[:,0],mod[:,1])
    
        F.append(f)
        k = k+1
    
    return list_ord,wl,data_tot,phase,window,Stdtot,SNRtot,projtot,F,Vc,berv

def interpolate_model(F,wl,Vtot,pixel_window,weights):
    """
    Interpolate model spectra for each order over a range of velocity shifts.

    Parameters:
    F (list of callable): List of interpolation functions for model spectra.
    wl (list of ndarray): List of wavelength arrays for each order.
    Vtot (ndarray): Array of velocity shifts.
    
    Global parameters:
    pixel_window (float): Width of the pixel for averaging.
    weights (ndarray): Weighting factors for averaging.
    list_ord (list): List of order numbers.

    Returns:
    F2D (list): List of 1D interpolation functions for shifted model spectra.
    """
    F2D = []
    for i in range(len(wl)):
        mod_int = np.zeros((len(Vtot),len(wl[i])))
        for j in range(len(Vtot)):
            #shift the wavelength
            #average on a pixel size
            mod_int[j]= np.average(F[i](list(map(lambda x: wl[i]/(1.0+(Vtot[j]+x)/(c0/1000.)),pixel_window))),weights=weights,axis=0)
        f2D = interp.interp1d(Vtot,mod_int.T)
        F2D.append(f2D)
        print("interp finished for Order ", i+1, "/", len(wl))
        
    return F2D


def interpolate_model_parallel(F_proc,wl_proc,Vtot,pixel_window,weights):
    """
    Interpolate model spectra in parallel for multiple orders. The difference with the 
    sequential function is that the interpolation is just done over the orders of interest

    Parameters:
    F_proc (list of callable): List of interpolation functions for model spectra.
    wl_proc (list of ndarray): List of wavelength arrays for each order.
    Vtot (ndarray): Array of velocity shifts.
    
    Global parameters:
    pixel_window (float): Width of the pixel for averaging.
    weights (ndarray): Weighting factors for averaging.

    Returns:
    F2D (list): List of 1D interpolation functions for shifted model spectra.
    """
    F2D = []
    for i in range(len(wl_proc)):
        mod_int = np.zeros((len(Vtot),len(wl_proc[i])))
        for j in range(len(Vtot)):
            #shift the wavelength
            #average on a pixel size
            mod_int[j]= np.average(F_proc[i](list(map(lambda x: wl_proc[i]/(1.0+(Vtot[j]+x)/(c0/1000.)),pixel_window))),weights=weights,axis=0)
        f2D = interp.interp1d(Vtot,mod_int.T,kind='linear')
        F2D.append(f2D)
        print("interp advancement:",(i+1)/len(wl_proc)*100, "%")
        
    return F2D


def perform_correlation(list_ord,data_tot,projtot,Stdtot,SNRtot,F2D,phase2,window2,Vstar,pos,Kp,Vsys,nbor,\
                        use_proj,proj_fast,mode_norm_pca,speed_planet,ecc,wp):

    """
    Perform correlation analysis between data and model spectra.
    
    Parameters:
    list_ord (list): List of order numbers.
    data_tot (list of ndarray): List of data arrays for each order.
    projtot (list of ndarray): List of projection arrays.
    Stdtot (list of ndarray): List of standard deviation arrays.
    SNRtot (list of ndarray): List of signal-to-noise ratio arrays.
    F2D (list of callable): List of 2D interpolation functions for model spectra.
    phase2 (ndarray): Array of phase values.
    window2 (ndarray): Windowing function.
    Vstar (ndarray): Stellar velocity.
    Kp (ndarray): Orbital semi-amplitude.
    Vsys (ndarray): Systemic velocity.
    pos (ndarray): Array of positions.
    
    Global parameters:
    nbor (int): Number of border pixels to exclude.
    use_proj (bool): Whether to use projection.
    proj_fast (bool): Fast projection mode.
    mode_norm_pca (str): PCA normalization mode.
    
    Returns:
    correl_boucher (ndarray:) Correlation results.
    """
    Nkp = len(Kp)
    Nv = len(Vsys)
    correl_boucher= np.zeros((Nkp,Nv,len(list_ord),len(phase2)))
    
    
    
    for no in range(len(list_ord)):
        dataij = (data_tot[no][pos][:,nbor:-nbor].T-np.mean(data_tot[no][pos][:,nbor:-nbor].T,axis=0))
        tosum = 1./np.mean(SNRtot[no][pos]**2)/Stdtot[no][nbor:-nbor]**2
        projo = projtot[no]
        
        for i in range(Nkp):
            for j in range(Nv):
                if use_proj:
                    if proj_fast:
                        
                        interpmod = (F2D[no](speed_planet(phase2,Kp[i],wp,ecc)+Vsys[j]+Vstar[pos]))*window2
                        interpmod_fin = (np.exp(np.log(interpmod+1) - np.matmul(projo[pos][:,pos[0]],np.log(interpmod+1).T).T))
                    else:
                        # #projector in different cases. The "none" and "global" cases are actually
                        #exactly similar to the proj_fast, but much slower if you go through here
                        interpmod = np.zeros(data_tot[no].shape)
                        interpmod[pos] = (F2D[no](speed_planet(phase2,Kp[i],wp,ecc)+Vsys[j]+Vstar[pos])*window2).T
                        Il    = np.log(interpmod+1.0) #just to avoid nans

                        if mode_norm_pca =="none":
                            ff = Il
                            
                            im = 0.
                            ist = 1.
                            
                        elif mode_norm_pca =="global":
                            im            = np.nanmean(Il)
                            ist           = np.nanstd(Il)   
                            ff            = (Il - im)/ist     
                            
                        elif mode_norm_pca == "per_pix":

                            im = np.tile(np.nanmean(Il,axis=0),(Il.shape[0],1))
                            ist = np.tile(np.nanstd(Il,axis=0),(Il.shape[0],1))
                            
                            ff = (Il-im)/ist
                        
                        elif mode_norm_pca == "per_obs":

                            # we just have to be careful here as some lines are identically 0
                            im = np.tile(np.nanmean(Il,axis=1),(Il.shape[1],1)).T
                            ist = np.tile(np.nanstd(Il,axis=1),(Il.shape[1],1)).T
                            
                            ff = np.zeros(Il.shape)
                            ff[pos] = (Il[pos]-im[pos])/ist[pos]  
                            
                        model_ret = ff-np.matmul(projo,ff)
                        toexp = (model_ret*ist+im)[pos].T
                        # toexp = (model_ret)[pos].T

                        interpmod_fin= (np.exp(toexp)-1.0)

                else:
                    interpmod_fin = F2D[no](speed_planet(phase2,Kp[i],wp,ecc)+Vsys[j]+Vstar[pos])*window2

    
                modelij = interpmod_fin[nbor:-nbor]-\
                        np.mean(interpmod_fin[nbor:-nbor],axis=0)
                correl_boucher[i,j,no] = np.sum(((dataij*modelij*SNRtot[no][pos]**2).T*tosum).T,axis=0)#*np.mean(OBS.snr)*np.shape(modelij)[0]

        print(list_ord[no])
    return correl_boucher


def plot_correlation(list_ord,correl_boucher,select_plot,lili,Kp,Vsys,\
                     Kp_min_std,Kp_max_std,Vsys_min_std,Vsys_max_std,nlevels,white_lines=False,Kp_planet=0.0,Vsys_planet=0.0):
    """
    Plot correlation results.

    Parameters:

    correl_boucher (ndarray): Correlation results.
    list_ord (list): List of order numbers.

    
    Global parameters:
    Kp (ndarray): Orbital semi-amplitude.
    Vsys (ndarray): Systemic velocity.
    
    lili (list or ndarray, optional): List of orders to plot correlation. 
    select_plot (boolean): Do we use lili or list_ord ?
    Kp_min_std (float): Minimum orbital semi-amplitude for standard deviation mask.
    Kp_max_std (float): Maximum orbital semi-amplitude for standard deviation mask.
    Vsys_min_std (float): Minimum systemic velocity for standard deviation mask.
    Vsys_max_std (float): Maximum systemic velocity for standard deviation mask.
    Vsys_planet (float): Systemic velocity of the planet.
    Kpmin (float): Minimum orbital semi-amplitude for lines.
    Kpmax (float): Maximum orbital semi-amplitude for lines.
    Vmin (float): Minimum systemic velocity for lines.
    Vmax (float): Maximum systemic velocity for lines.
    Kp_planet (float): Orbital semi-amplitude of the planet.
    white_lines (bool, optional): Whether to plot white lines at the position of the planet.
    nlevels (int): Number of contour levels for the plot.
    """ 
    Nkp = len(Kp)
    Nv = len(Vsys)
    Kpmin = np.min(Kp)
    Kpmax = np.max(Kp)
    Vmin = np.min(Vsys)
    Vmax = np.max(Vsys)
    sel = []
    if select_plot:
        k=-1
        for no in lili:
            k+=1
            try:
                sel = sel +[ np.where(list_ord==no)[0][0]]
            except:
                continue
        correl_summed = np.sum(np.sum(correl_boucher[:,:,sel],axis=3),axis=2)
    else:
        correl_summed = np.sum(np.sum(correl_boucher,axis=3),axis=2)

    
    
    mask_std = np.ones((Nkp,Nv),dtype=bool)
    for i in range(Nkp):
        for j in range(Nv):
            if ((Kp[i]>Kp_min_std and Kp[i]<Kp_max_std) and (Vsys[j]>Vsys_min_std and Vsys[j]<Vsys_max_std)):
                mask_std[i,j] = False
    c = correl_summed[mask_std]
    snrmap = np.std((c))
    
    #
    plt.figure()
    
    # Same as above: uncomment first or second line
    
    plt.contourf(Vsys,Kp,correl_summed/snrmap,cmap="gist_heat",levels=nlevels)
    plt.colorbar(label="SNR")

    nsigma = np.max(correl_summed/snrmap)
    plt.contour(Vsys,Kp,correl_summed/snrmap,linestyles=["solid","dotted","dashed"],levels=[nsigma-3,nsigma-2,nsigma-1,nsigma],colors="white")

        

    plt.xlabel("Doppler shift (km.s$^{-1}$)")
    plt.ylabel("Orbital semi amplitude (km.s$^{-1}$)")
    
    if white_lines:
        plt.plot([Vsys_planet,Vsys_planet],[Kpmin,Kpmax],'--',color='white')
        plt.plot([Vmin,Vmax],[Kp_planet,Kp_planet],'--',color='white')
        
    plt.show()
            


  
def plot_correlation_tot(list_tot,correl_tot,select_plot,lili,Kp,Vsys,\
                     Kp_min_std,Kp_max_std,Vsys_min_std,Vsys_max_std,nlevels,white_lines=False,Kp_planet=0.0,Vsys_planet=0.0):
    """
    Plot correlation results.

    Parameters:

    correl_boucher (ndarray): Correlation results.
    list_ord (list): List of order numbers.

    
    Global parameters:
    Kp (ndarray): Orbital semi-amplitude.
    Vsys (ndarray): Systemic velocity.
    
    lili (list or ndarray, optional): List of orders to plot correlation. 
    select_plot (boolean): Do we use lili or list_ord ?
    Kp_min_std (float): Minimum orbital semi-amplitude for standard deviation mask.
    Kp_max_std (float): Maximum orbital semi-amplitude for standard deviation mask.
    Vsys_min_std (float): Minimum systemic velocity for standard deviation mask.
    Vsys_max_std (float): Maximum systemic velocity for standard deviation mask.
    Vsys_planet (float): Systemic velocity of the planet.
    Kpmin (float): Minimum orbital semi-amplitude for lines.
    Kpmax (float): Maximum orbital semi-amplitude for lines.
    Vmin (float): Minimum systemic velocity for lines.
    Vmax (float): Maximum systemic velocity for lines.
    Kp_planet (float): Orbital semi-amplitude of the planet.
    white_lines (bool, optional): Whether to plot white lines at the position of the planet.
    nlevels (int): Number of contour levels for the plot.
    """ 
    Nkp = len(Kp)
    Nv = len(Vsys)
    Kpmin = np.min(Kp)
    Kpmax = np.max(Kp)
    Vmin = np.min(Vsys)
    Vmax = np.max(Vsys)
    correl_summed = np.zeros((Nkp,Nv))

    for i in range(len(correl_tot)):
        sel = []
        if select_plot:
            k=-1
            for no in lili:
                k+=1
                try:
                    sel = sel +[ np.where(list_tot[i]==no)[0][0]]
                except:
                    continue
            correl_summed += np.sum(np.sum(correl_tot[i][:,:,sel],axis=3),axis=2)
        else:
            correl_summed += np.sum(np.sum(correl_tot[i],axis=3),axis=2)

    
    
    mask_std = np.ones((Nkp,Nv),dtype=bool)
    for i in range(Nkp):
        for j in range(Nv):
            if ((Kp[i]>Kp_min_std and Kp[i]<Kp_max_std) and (Vsys[j]>Vsys_min_std and Vsys[j]<Vsys_max_std)):
                mask_std[i,j] = False
    c = correl_summed[mask_std]
    snrmap = np.std((c))
    
    #
    plt.figure()
    
    # Same as above: uncomment first or second line
    
    plt.contourf(Vsys,Kp,correl_summed/snrmap,cmap="gist_heat",levels=nlevels)
    plt.colorbar(label="SNR")

    nsigma = np.max(correl_summed/snrmap)
    plt.contour(Vsys,Kp,correl_summed/snrmap,linestyles=["solid","dotted","dashed"],levels=[nsigma-3,nsigma-2,nsigma-1,nsigma],colors="white")

        

    plt.xlabel("Doppler shift (km.s$^{-1}$)")
    plt.ylabel("Orbital semi amplitude (km.s$^{-1}$)")
    
    if white_lines:
        plt.plot([Vsys_planet,Vsys_planet],[Kpmin,Kpmax],'--',color='white')
        plt.plot([Vmin,Vmax],[Kp_planet,Kp_planet],'--',color='white')
        
    plt.show()
            
      
        


