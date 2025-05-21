#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/2022

@author: Baptiste & Florian
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker
import warnings
import matplotlib.cbook

from scipy.optimize import minimize




# -----------------------------------------------------------
# Compare the dispersion at the center of each spectrum (in 
# each order) to the photon noise provided by the SPIRou DRS
# -----------------------------------------------------------
def plot_spectrum_dispersion(lord,nam_fig):

    """
    --> Inputs:     - lord: list of Order objects

    --> Outputs:    - Plot displayed

     """

    # Initialization
    rms_sp     = np.zeros(len(lord))
    rms_sp_s   = np.zeros(len(lord))
    rms_drs    = np.zeros(len(lord))
    rms_drs_s  = np.zeros(len(lord))
    rms_pca    = np.zeros(len(lord))
    rms_pca_s  = np.zeros(len(lord))    
    wmean      = np.zeros(len(lord))
    LO         = np.zeros(len(lord),dtype=int)

    for kk in range(len(lord)):
        O              = lord[kk]
        disp_mes       = 1./O.SNR_mes
        disp_drs       = 1./O.SNR
        disp_pca       = 1./O.SNR_mes_pca
        rms_sp[kk]     = np.mean(disp_mes)
        rms_sp_s[kk]   = np.std(disp_mes)
        rms_drs[kk]    = np.mean(disp_drs)
        rms_drs_s[kk]  = np.std(disp_drs)
        rms_pca[kk]    = np.mean(disp_pca)
        rms_pca_s[kk]  = np.std(disp_pca)
        wmean[kk]      = O.W_mean
        LO[kk]         = O.number

    # Compute wavelength-order number correspondance
    WW,LO_pred,LO_predt = fit_order_wave(LO,wmean)
    plt.figure(figsize=(12,5))
    ax = plt.subplot(111)
    ax.errorbar(LO,rms_sp,rms_sp_s,fmt="*",color="k",label="Reduced data",capsize=10.0,ms=10.)
    ax.errorbar(LO,rms_pca,rms_pca_s,fmt="^",color="g",label="After PCA",capsize=10.0,ms=7.5)
    ax.errorbar(LO,rms_drs,rms_drs_s,fmt="o",color="m",label="DRS",capsize=8.0)
    
    ax.legend(ncol=2)
    ax2 = ax.twiny()
    ax2.set_xticks(LO_pred)
    ax2.set_xlabel("Wavelength [nm]")
    ax2.set_xticklabels(WW)
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(LO_predt))
    ax.set_xlim(30,80)
    ax2.set_xlim(30,80)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_ylabel("Spectrum dispersion")
    ax.set_xlabel("Order number")
    ax.set_yscale("log")
    plt.subplots_adjust(wspace=0.5,hspace = 0.)
    plt.savefig(nam_fig,bbox_inches="tight")
    plt.close()
    




def plot_reduction(phase,W1,I1,W2,I2,W3,I3,W4,I4,lab=["1","2","3","4"],filenam="reduc.png",lmin=-1,lmax=-1):



    ### Show the evolution of the sequence of spectra for a given order
    cmap = "gist_earth" # Another 'fancy' color map?


    plt.figure(figsize=(15,15))

    ###############################        
    ax   = plt.subplot(411)
    X,Y  = np.meshgrid(W1,phase)
    Z    = I1
    zmin = I1.mean() - 3.*I1.std()
    zmax = I1.mean() + 3.*I1.std()
    c    = plt.pcolor(X,Y,Z,cmap=cmap)#,vmin=zmin,vmax=zmax) 
    plt.colorbar(c,ax=ax)
    ax.set_ylabel("Orbital phase")  
    ax.set_xticks([])        
    if lmin>-1: ax.set_xlim(lmin,lmax)
    else: ax.set_xlim(W4.min(),W4.max())

    tx = lab[0]                           
    plt.text(np.min(W4)+1.5,0.8*np.max(phase),tx,color="w",fontsize=20,fontweight="bold")

    ###############################  
    ax   = plt.subplot(412)
    X,Y  = np.meshgrid(W2,phase)
    Z    = I2
    zmin = I2.mean() - 3.*I2.std()
    zmax = I2.mean() + 3.*I2.std()
    c    = plt.pcolor(X,Y,Z,cmap=cmap)#,vmin=zmin,vmax=zmax)   
    ax.set_ylabel("Orbital phase")  
    ax.set_xticks([])        
    if lmin>-1: ax.set_xlim(lmin,lmax)
    else: ax.set_xlim(W4.min(),W4.max())
    
    plt.colorbar(c,ax=ax)

    tx = lab[1]                           
    plt.text(np.min(W4)+1.5,0.8*np.max(phase),tx,color="w",fontsize=20,fontweight="bold")


    ###############################  
    ax   = plt.subplot(413)
    X,Y  = np.meshgrid(W3,phase)
    Z    = I3
    zmin = I3.mean() - 3.*I3.std()
    zmax = I3.mean() + 3.*I3.std()
    c    = plt.pcolor(X,Y,Z,cmap=cmap)#,vmin=zmin,vmax=zmax)  
    ax.set_ylabel("Orbital phase")  
    ax.set_xticks([])        
    if lmin>-1: ax.set_xlim(lmin,lmax)
    else: ax.set_xlim(W4.min(),W4.max())
    
    plt.colorbar(c,ax=ax)
    tx = lab[2]                           
    plt.text(np.min(W4)+1.5,0.8*np.max(phase),tx,color="w",fontsize=20,fontweight="bold")


    ###############################  
    ax   = plt.subplot(414)
    X,Y  = np.meshgrid(W4,phase)
    Z    = I4
    zmin = I4.mean() - 3.*I4.std()
    zmax = I4.mean() + 3.*I4.std()
    c    = plt.pcolor(X,Y,Z,cmap=cmap)#,vmin=zmin,vmax=zmax)  
    ax.set_ylabel("Orbital phase")  
    if lmin>-1: ax.set_xlim(lmin,lmax)
    else: ax.set_xlim(W4.min(),W4.max())
    
    plt.colorbar(c,ax=ax)
    ax.set_xlabel("Wavelength [nm]")

    tx = lab[3]                           
    plt.text(np.min(W4)+1.5,0.8*np.max(phase),tx,color="w",fontsize=20,fontweight="bold")
    plt.subplots_adjust(hspace=0.02)
    
    plt.savefig(filenam,bbox_inches="tight")
    #plt.show()
    plt.close()


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
# Simple hyperbola
# -----------------------------------------------------------
def hyp(par,xx):
    return par[0]/xx + par[1]

# --------------------
# Simple inverse hyperbola
# -----------------------------------------------------------
def hyp_inv(par,yy):
    return par[0]/(yy-par[1])

# ----------------------------------# -----------------------------------------------------------
# Return least-square difference between a hyperbola for 'par' 
# parameters and data yy.
# xx is the X-axis vector 
# -----------------------------------------------------------
def crit_hyp(par,xx,yy):
    y_pred = hyp(par,xx)
    return np.sum((yy-y_pred)**(2))  
    