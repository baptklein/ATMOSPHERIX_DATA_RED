#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:07:36 2021

@author: florian
"""

import numpy as np
import os
import convolve_templates as conv

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))

nf = 501


R_unit = "cm"
# R_unit = "none"
Rs = 0.777*696340000.0 # in solar unit

lambdas_unit = "micron"
lambdas_unit = "nano"

transit_depth = True
# transit_depth = False

broadening = True
broadening = False

norm = True
# norm = False

Nphase = 1
#name only needed if Nphase =1
name ="spectrum_hd189_exeter_h2o_aq803_harada_wr_exomol"

dire = "/home/florian/Bureau/Atmosphere_SPIRou/Models/HD189/Results/"
dire_res = dire+"/to-correl/"

# dire_res = "/home/florian/Bureau/Atmosphere_SPIRou/Models/GL15A/HD189/to-correl/"

#SPIRou
list_ord = np.arange(31,80)
wlen_file ="/home/florian/Bureau/Atmosphere_SPIRou/wlen.dat"




# #MAROON X
# wlen_file ="/home/florian/Bureau/Atmosphere_SPIRou/wlen_MAROONX.dat"
# list_ord = np.arange(67,125)
# list_ord = np.delete(list_ord,np.where(list_ord==80))
# list_ord = np.delete(list_ord,np.where(list_ord==81))

# list_ord = np.array([78])
    # list_ord=[33]

if transit_depth:
    tot = np.loadtxt(dire+name+".dat")
    fr = tot[:,0]*1000
    R = tot[:,1]


if Nphase == 1:
    if not transit_depth:
        fr = np.loadtxt(dire+"lambdas"+name+".txt")
        R = np.loadtxt(dire+"Rp"+name+".txt")
        if R_unit == "cm":
            R = R/100.

    wlen = np.loadtxt(wlen_file)

    if lambdas_unit=="micron":
        fr = fr*1000

    dire_resphase =dire_res+"reduced"+name

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
        
        lambdas = fr[np.where((fr>lmin*0.95) & (fr<lmax*1.05))]
        R_ord = R[np.where((fr>lmin*0.95) & (fr<lmax*1.05))]

        out = np.zeros((2,np.shape(lambdas)[0]))
        
        for j in range(np.shape(lambdas)[0]) :
            out[0,j] = lambdas[j]
            if transit_depth:
                out[1,j] = R_ord[j]
            else:
                out[1,j] = (R_ord[j]/Rs)**2
                
        if broadening:
            conv_wl,conv_R = conv.rotate( out[1],out[0],0.0,0.0)
            out = np.zeros((2,len(conv_wl)))
            out[0] = conv_wl
            out[1] = conv_R
                

        if norm:
            out_final = np.zeros((2,len(out[1])-nf+1))
            try :
                win = np.percentile(strided_app(out[1],nf,1),0.5, axis=-1)
            except:
                print ("Removed order : ",list_ord[no])
                continue
            out_final[0]  =out[0][int((nf-1)/2):-int((nf-1)/2)]
            out_final[1] = win - out[1][int((nf-1)/2):-int((nf-1)/2)]
            
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
        

else:
    if lambdas_unit=="micron":
        fr = np.loadtxt(dire+"lambdas0.txt")*1000
    else:
        fr = np.loadtxt(dire+"lambdas0.txt")
    wlen = np.loadtxt(wlen_file)
    for i in range(Nphase):
        print(i)
        R = np.loadtxt(dire+"Rp"+str(i)+".txt")
        dire_resphase =dire_res+str(i)+"/"
        try :
            os.mkdir(dire_resphase)
        except:
            print("phase folder already exists")
        for no in range(len(list_ord)):
            lmin = wlen[list_ord[no]-31,1]
            lmax = wlen[list_ord[no]-31,2]

            lambdas = fr[np.where((fr>lmin*0.99) & (fr<lmax*1.01))]
            R_ord = R[np.where((fr>lmin*0.99) & (fr<lmax*1.01))]

            out = np.zeros((2,np.shape(lambdas)[0]))

            for j in range(np.shape(lambdas)[0]) :
                out[0,j] = lambdas[j]
                out[1,j] = 1.0-(R_ord[j]/Rs)**2

            len_conv = 1000
            mova = np.convolve(out[1],np.ones(len_conv)/len_conv,mode="valid")
            out[1,len_conv-1:] = out[1,len_conv-1:]-mova
            # out[1,len_conv-1:] = out[1,len_conv-1:]
            out[1,len_conv-1:] -= np.max(out[1,len_conv-1:])
            np.savetxt(dire_resphase+"template"+str(list_ord[no])+".txt",out[:,len_conv-1:].T)



















