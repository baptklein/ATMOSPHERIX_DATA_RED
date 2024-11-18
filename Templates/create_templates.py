#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:07:36 2021

@author: florian
"""

import numpy as np
import os
import convolve_templates as conv


c0 = 299792458.0
G = 6.67e-11
h_planck =  6.62607015e-34
k_boltzmann = 1.380649e-23

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

def B(lambdas,T): #Planck function as a function of wavelength for emission spectroscopy

    return 2*h_planck*c0**2/(lambdas**5)/(np.exp(h_planck*c0/lambdas/T/k_boltzmann)-1.)
nf = 501

type_templ = "tranmission"
type_templ = "emission"

R_unit = "cm"
# R_unit = "none"
Rs = 1.347*696340000.0 # in solar unit
Ts = 3500.

Rp = 1.0*69911000.0

lambdas_unit = "micron"
lambdas_unit = "nano"


transit_depth = True
transit_depth = False

broadening = True
broadening = False

norm = True
# norm = False

Nphase = 1
#name only needed if Nphase =1
name ="test-GL15A_onlyH2O"

dire = "/home/adminloc/Bureau/Atmospheres/Pipeline_v2/ATMOSPHERIX_DATA_RED/Models/Results/"
dire_res = "/home/adminloc/Bureau/Atmospheres/Pipeline_v2/ATMOSPHERIX_DATA_RED/Templates/"
# dire_res = "/home/florian/Bureau/Atmosphere_SPIRou/Models/GL15A/HD189/to-correl/"

#SPIRou
list_ord = np.arange(31,80)
wlen_file ="/home/adminloc/Bureau/Atmospheres/Pipeline_v2/ATMOSPHERIX_DATA_RED/wlen.dat"




# #MAROON X
# wlen_file ="/home/florian/Bureau/Atmosphere_SPIRou/wlen_MAROONX.dat"
# list_ord = np.arange(67,125)
# list_ord = np.delete(list_ord,np.where(list_ord==80))
# list_ord = np.delete(list_ord,np.where(list_ord==81))

# list_ord = np.array([78])
    # list_ord=[33]


if type_templ =="transmission":
    if transit_depth:
        tot = np.loadtxt(dire+name+".dat")
        wavelength = tot[:,0]*1000
        planet = tot[:,1]
    
    
    if not transit_depth:
        wavelength = np.loadtxt(dire+"lambdas"+name+".txt")
        planet = np.loadtxt(dire+"Rp"+name+".txt")
        
        if R_unit == "cm":
            planet = planet/100.
else:
    wavelength = np.loadtxt(dire+"lambdas"+name+".txt")
    planet = np.loadtxt(dire+"flux"+name+".txt")

wlen = np.loadtxt(wlen_file)

if lambdas_unit=="micron":
    wavelength = wavelength*1000

dire_resphase =dire_res+name

if norm:
    try :
        os.mkdir(dire_resphase)
    except:
        print("phase folder already exist")
else:
    try :
        os.mkdir(dire_resphase+"_nonorm/")
    except:
        print("phase folder already exist")


for i in range(len(list_ord)):
    no=np.where(wlen[:,0]==list_ord[i])[0][0]
    lmin = wlen[no,1]
    lmax = wlen[no,2]
    
    lambdas = wavelength[np.where((wavelength>lmin*0.95) & (wavelength<lmax*1.05))]
    planet_ord = planet[np.where((wavelength>lmin*0.95) & (wavelength<lmax*1.05))]

    out = np.zeros((2,np.shape(lambdas)[0]))
    
    for j in range(np.shape(lambdas)[0]) :
        out[0,j] = lambdas[j]
        if type_templ=="tranmission":
            if transit_depth:
                out[1,j] = planet_ord[j]
            else:
                out[1,j] = (planet_ord[j]/Rs)**2
        else:
            out[1,j] = planet_ord[j]*(Rp/Rs)**2/np.pi/B(out[0,j]*1e-9,Ts) #We assume here that the Doppler shift is not important
            # for the stellar spectrum .... that's a trouble with RM
            
    if broadening:
        conv_wl,conv_planet = conv.rotate( out[1],out[0],0.0,0.0)
        out = np.zeros((2,len(conv_wl)))
        out[0] = conv_wl
        out[1] = conv_planet
            

    if norm:
        out_final = np.zeros((2,len(out[1])-nf+1))
        try :
            win = np.percentile(strided_app(out[1],nf,1),0.5, axis=-1)
        except:
            print ("Removed order : ",list_ord[no])
            continue
        out_final[0]  =out[0][int((nf-1)/2):-int((nf-1)/2)]
        if type_templ == "transmission":
            out_final[1] = win - out[1][int((nf-1)/2):-int((nf-1)/2)]
        else:
            out_final[1] = out[1][int((nf-1)/2):-int((nf-1)/2)]-win
        
        if broadening:
            np.savetxt(dire_resphase+"/"+"templatebroad"+str(list_ord[i])+".txt",out_final.T)
        else:
            np.savetxt(dire_resphase+"/"+"template"+str(list_ord[i])+".txt",out_final.T)


    else:
        out_final = np.zeros((2,len(out[1])))

        out_final[0]  =out[0]
        out_final[1] = out[1]
        
        if broadening:
            np.savetxt(dire_resphase+"_nonorm/"+"templatebroad"+str(list_ord[i])+".txt",out_final.T)
        else:
            np.savetxt(dire_resphase+"_nonorm/"+"template"+str(list_ord[i])+".txt",out_final.T)

    # out_final[1] = -(out[1][int((nf-1)/2):-int((nf-1)/2)]-np.mean(out[1][int((nf-1)/2):-int((nf-1)/2)]))
    