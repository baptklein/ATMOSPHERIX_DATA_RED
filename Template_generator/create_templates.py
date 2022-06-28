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
Rs = 0.375*696340000.0 # in solar unit
lambdas_unit = "micron"
lambdas_unit = "nano"
Nphase = 1
#name only needed if Nphase =1
name ="GL15A_HD189"


dire = "/home/florian/Bureau/Atmosphere_SPIRou/Models/GL15A/HD189/"
dire_res = "/home/florian/Bureau/Atmosphere_SPIRou/Models/GL15A/HD189/"


#SPIRou
list_ord = np.arange(31,80)
wlen_file ="/home/florian/Bureau/Atmosphere_SPIRou/wlen_orders.dat"




# #MAROON X
# wlen_file ="/home/florian/Bureau/Atmosphere_SPIRou/wlen_MAROONX.dat"
# list_ord = np.arange(67,125)
# list_ord = np.delete(list_ord,np.where(list_ord==80))
# list_ord = np.delete(list_ord,np.where(list_ord==81))

# list_ord = np.array([58])
    # list_ord=[33]

rotation = True
# rotation  = False

vrot = 3522.96
superrot=10000.0

if Nphase == 1:
    fr = np.loadtxt(dire+"lambdas"+name+".txt")
    if lambdas_unit=="micron":
        fr = fr*1000
    wlen = np.loadtxt(wlen_file)
    R = np.loadtxt(dire+"Rp"+name+".txt")
    if R_unit == "cm":
        R = R/100.
    
    # R = np.load(dire+"transit_depth.npy")/Rs**2
    
    if not rotation:
        
        R = (R/(Rs))**2
        dire_resphase =dire_res+"reduced"+name+"/"
        try :
            os.mkdir(dire_resphase)
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
                # out[1,j] = (R_ord[j]/Rs)**2
                out[1,j] = R_ord[j]

            win = np.percentile(strided_app(out[1],nf,1),0.5, axis=-1)
            try :
                out_final = np.zeros((2,len(out[1])-nf+1))
            except:
                print ("Removed order : ",list_ord[no])
                continue
            out_final[0]  =out[0][int((nf-1)/2):-int((nf-1)/2)]
            out_final[1] = win - out[1][int((nf-1)/2):-int((nf-1)/2)]
            
            np.savetxt(dire_resphase+"template"+str(list_ord[i])+".txt",out_final.T)
    else:

        dire_resphase =dire_res+"superrot10"+name+"/"
        try :
            os.mkdir(dire_resphase)
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
                out[1,j] = R_ord[j]
            conv_wl,conv_R1 = conv.rotate( out[1],out[0],vrot,superrot)
            conv_R = (conv_R1/Rs)**2
            win = np.percentile(strided_app(conv_R,nf,1),0.5, axis=-1)
            try :
                out_final = np.zeros((2,len(conv_R)-nf+1))
            except:
                print ("Removed order : ",list_ord[no])
                continue
            # out_final[0]  =conv_wl[int((nf-1)/2):-int((nf-1)/2)]
            # out_final[1] = win - conv_R[int((nf-1)/2):-int((nf-1)/2)]

            out_final[0]  =conv_wl[int((nf-1)/2):-int((nf-1)/2)]
            out_final[1] = conv_R1[int((nf-1)/2):-int((nf-1)/2)]
            
            np.savetxt(dire_resphase+"template"+str(list_ord[i])+".txt",out_final.T)

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



















