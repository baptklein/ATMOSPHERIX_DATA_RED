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

R_unit = "cm"
R_unit = "m"

lambdas_unit = "micron"
lambdas_unit = "nano"

suffix = '_broadened'

name = "GL15A_HD189_onlyH2O-VMR3-T900"



dire = "/home/florian/Bureau/Atmosphere_SPIRou/Pipeline_git/ATMOSPHERIX_DATA_RED/Data_Simulator/Model/Results/"
dire_res = dire

#SPIRou
list_ord = np.arange(31,80)
wlen_file ="/home/florian/Bureau/Atmosphere_SPIRou/wlen_orders.dat"


vrot = 100.0
superrot=0.0


fr = np.loadtxt(dire+"lambdas"+name+".txt")
R = np.loadtxt(dire+"Rp"+name+".txt")

if R_unit == "cm":
    R = R/100.

if lambdas_unit=="micron":
    fr = fr*1000

wlen = np.loadtxt(wlen_file)

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
   
    out_final = np.zeros((2,len(conv_R)))

    out_final[0]  =conv_wl
    out_final[1] = conv_R
    final_mean = np.mean(out_final[1])
    out_final[1] = out_final[1]-final_mean+init_mean

    #We have to be careful not to mix up everything 
    if i==0:
        wlen_final = out_final[0][::-1]
        R_final =  out_final[1][::-1]
        R_2 = out_final[1][::-1]+final_mean-init_mean
    else:
        R_final = np.append(out_final[1][np.where(out_final[0]<np.min(wlen_final))][::-1],R_final)
        R_2 = np.append(out_final[1][np.where(out_final[0]<np.min(wlen_final))][::-1]+final_mean-init_mean,R_2)
        wlen_final = np.append(out_final[0][np.where(out_final[0]<np.min(wlen_final))][::-1],wlen_final)
        
#The final data are in nm and cm
np.savetxt(dire+"lambdas"+name+suffix+".txt",wlen_final)
np.savetxt(dire+"Rp"+name+suffix+".txt",R_final*100)
        