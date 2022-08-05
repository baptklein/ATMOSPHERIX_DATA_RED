#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:46:04 2022

@author: florian
"""


import numpy as np
import os
import convolve_templates as conv
import matplotlib.pyplot as plt

SMALL_SIZE = 28
MEDIUM_SIZE = 32
BIGGER_SIZE = 34

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

nf = 501


R_unit = "cm"
Rs = 0.375*696340000.0 # in solar unit
lambdas_unit = "micron"
lambdas_unit = "nano"
Nphase = 1
#name only needed if Nphase =1
name ="GL15A_HD189_onlyH2O-VMR3-T900"
suffix = '_rotated3000-goodmean'


dire = "/home/florian/Bureau/Atmosphere_SPIRou/ATMOSPHERIX_DATA_RED/Data_Simulator/Model/Misc/"
dire_res = dire

#SPIRou
list_ord = np.arange(31,80)
wlen_file ="/home/florian/Bureau/Atmosphere_SPIRou/wlen_orders.dat"

# #MAROON X
# wlen_file ="/home/florian/Bureau/Atmosphere_SPIRou/wlen_MAROONX.dat"
# list_ord = np.arange(67,125)
# list_ord = np.delete(list_ord,np.where(list_ord==80))
# list_ord = np.delete(list_ord,np.where(list_ord==81))


vrot = 3000.0
superrot=0.0


if Nphase == 1:
    fr = np.loadtxt(dire+"lambdas"+name+".txt")
    if lambdas_unit=="micron":
        fr = fr*1000
    wlen = np.loadtxt(wlen_file)
    R = np.loadtxt(dire+"Rp"+name+".txt")
    if R_unit == "cm":
        R = R/100.

#I decide to rotate order by order but it could be done in another manner to be more precise
    for i in range(len(list_ord)):
        no=np.where(wlen[:,0]==list_ord[i])[0][0]
        lmin = wlen[no,1]
        lmax = wlen[no,2]
        
        lambdas = fr[np.where((fr>lmin*0.95) & (fr<lmax*1.05))]
        R_ord = R[np.where((fr>lmin*0.95) & (fr<lmax*1.05))]
        init_mean = np.mean(R_ord)

        out = np.zeros((2,np.shape(lambdas)[0]))
        
        for j in range(np.shape(lambdas)[0]) :
            out[0,j] = lambdas[j]
            out[1,j] = R_ord[j]
        conv_wl,conv_R = conv.rotate( out[1],out[0],vrot,superrot)
       
        # conv_R = (conv_R1/Rs)**2
        win = np.percentile(strided_app(out[1],nf,1),0.5, axis=-1)
        try :
            out_final = np.zeros((2,len(conv_R)-nf+1))
        except:
            print ("Removed order : ",list_ord[no])
            continue
        out_final[0]  =conv_wl[int((nf-1)/2):-int((nf-1)/2)]
        out_final[1] = conv_R[int((nf-1)/2):-int((nf-1)/2)]
        final_mean = np.mean(out_final[1])
        out_final[1] = out_final[1]-final_mean+init_mean

        if i==0:
            wlen_final = out_final[0][::-1]
            R_final =  out_final[1][::-1]
        else:
            R_final = np.append(out_final[1][np.where(out_final[0]<np.min(wlen_final))][::-1],R_final)
            wlen_final = np.append(out_final[0][np.where(out_final[0]<np.min(wlen_final))][::-1],wlen_final)
    np.savetxt(dire+"lambdas"+name+suffix+".txt",wlen_final)
    np.savetxt(dire+"Rp"+name+suffix+".txt",R_final*100)
            