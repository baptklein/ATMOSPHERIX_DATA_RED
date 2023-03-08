#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:54:11 2021

@author: florian
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.signal
from scipy import stats
import pickle
import time


pipeline_rep = "/home/florian/Bureau/Atmosphere_SPIRou/Pipeline_git/ATMOSPHERIX_DATA_RED/"

#WHere is your pkl file and what is its name. don't forget the slash at the end of the directrory
REP_DATA = pipeline_rep
filename  = 'Simu_GL15A_HD189_Kp120_v30_onlyH2O-VMR3-T900_reduced.pkl' 

#Directory of your reduced model
dire_mod = pipeline_rep+"Data_Simulator/Model/Results/to-correl/reducedGL15A_HD189_onlyH2O-VMR3-T900/"

#Kp intervals and size
Kpmin = 20.0
Kpmax =220.0
Nkp = 21
Kp = np.linspace(Kpmin,Kpmax,Nkp)



#Vsys intervals and size
Vmin = 0
Vmax= 60
Nv = 31
Vsys = np.linspace(Vmin,Vmax,Nv)



#This is just for the integration over a pixel
c0 = 299792.458
pixel_window = np.linspace(-1.17,1.17,15)
weights = scipy.signal.gaussian(15,std=1.17)
weights= np.ones(15)


wl = []
data_tot=[]
F =[]
Stdtot = []
SNRtot = []
projtot = []


with open(REP_DATA+filename,'rb') as ccfile:
    orders,W_data,I_data,T_obs,phase,window,berv,Vc,airmass,SN,proj = pickle.load(ccfile)


#choose the orders to consider. SOmetime there is a problem with the last one
list_ord = np.array(orders)
list_ord = np.array(orders[:-1])

#The data directory contains the data after reduction
#THe mod directory contains the templates in ntwo columns : wl and (1-rp/Rs**2) normalised
k  = 0
for no in orders:
    if no in list_ord:
        pass
    else:
        k=k+1
        continue

    file_mod = dire_mod+"/template"+str(no)+".txt"

    try:
        mod = np.loadtxt(file_mod)
    except:
        print("Careful ! Headers on model. Assuming 5 lines")
        mod = np.loadtxt(file_mod,skiprows=5)
        

    Std = np.zeros(np.shape(I_data[k])[1])
    for i in range(np.shape(I_data[k])[1]):
        Std[i] = (np.std(I_data[k][:,i]))
    
    #if there are issues in the data reduction we discard the orders
    try:
        fit2 = np.polyfit(W_data[k],Std,2)
    except:
        list_ord = np.delete(list_ord,k)
        k=k+1
        continue
    # Stdfit = np.poly1d(fit2)
    
    #Decide whether tou use the actual Std or the polynomial fit 
    #of the Std (see footnote in the paper)
    # Stdtot.append(Stdfit(OBS.W_raw))
    Stdtot.append(Std)
    
    
    data_tot.append(I_data[k])
    wl.append(W_data[k])
    


    SNRtot.append(SN[k])
    projtot.append(proj[k])

    #prepare model interpolation
    f = interp.interp1d(mod[:,0],mod[:,1])

    F.append(f)
    k = k+1



#This is the maxi speed up of the code. We create an interpolation
#of the model as a function of speed, integrated ovver a
# pixel size. We just need to call this function afterwards instead of integrating
Vstarmax = np.max(np.abs(np.array(Vc)-np.array(berv)))
Vint_max = np.max(Kpmax*np.abs(np.sin(2.0*np.pi*np.array(phase))))+np.max(np.abs([Vmin,Vmax]))+Vstarmax
Vtot = np.linspace(-1.01*Vint_max,1.01*Vint_max,8*int(Vint_max))
F2D = []
for i in range(len(list_ord)):
    mod_int = np.zeros((len(Vtot),len(wl[i])))
    for j in range(len(Vtot)):
        #shift the wavelength
        #average on a pixel size
        mod_int[j]= np.average(F[i](list(map(lambda x: wl[i]/(1.0+(Vtot[j]+x)/c0),pixel_window))),weights=weights,axis=0)
    f2D = interp.interp1d(Vtot,mod_int.T,kind='linear')
    F2D.append(f2D)
    print("interp finished for Order ", list_ord[i])

print("let's correl")
#And now the correlation, following boucher
#We only keep where window is >0 for speeding up the code
pos = np.where(np.array(window)>0.0)
phase2 = np.array(phase)[pos]
window2 = np.array(window)[pos]
correl_boucher= np.zeros((Nkp,Nv,len(list_ord),len(phase2)))



Vstar = np.array(Vc)-np.array(berv)
Vsatr2 = Vstar[pos]

nbor = 50
start = time.time()
for no in range(len(list_ord)):
    dataij = (data_tot[no][pos][:,nbor:-nbor].T-np.mean(data_tot[no][pos][:,nbor:-nbor].T,axis=0))
    tosum = np.mean(SNRtot[no][pos]**2)/Stdtot[no][nbor:-nbor]**2

    projo = projtot[no]


    for i in range(Nkp):
        for j in range(Nv):

            # 
            # uncomment if not using proj
            # interpmod = F2D[no](Kp[i]*np.sin(2.0*np.pi*np.array(phase2))+Vsys[j]+Vstar[pos])*window2
            #########################"
            
            # uncomment if using standard projector (no column mean, as thea)
            interpmod = F2D[no](Kp[i]*np.sin(2.0*np.pi*np.array(phase2))+Vsys[j]+Vstar[pos])*window2
            interpmod = (np.exp(np.log(interpmod+1) - np.matmul(projo[pos][:,pos[0]],np.log(interpmod+1).T).T) - 1)
            ##########################
            
            # uncomment for fancy projector 
            # interpmod = np.zeros(data_tot[no].shape)
            # interpmod[pos] = (F2D[no](Kp[i]*np.sin(2.0*np.pi*np.array(phase2))+Vsys[j]+Vstar[pos])*window2).T
            # Il    = np.log(interpmod+1.0) #just to avoid nans

            
            # im = np.tile(np.nanmean(Il,axis=0),(len(phase),1))
            # ist = np.tile(np.nanstd(Il,axis=0),(len(phase),1))

            
            # ff = (Il-im)/ist


            # # #Option 1
            # model_ret = ff-np.matmul(projo,ff)
            
            # toexp = model_ret[pos]*ist+im

            # model_ret= (np.exp(toexp)-1.0)

            # interpmod = model_ret.T
            #############################""
            
            
            


            modelij = interpmod[nbor:-nbor]-\
                    np.mean(interpmod[nbor:-nbor],axis=0)
  
            correl_boucher[i,j,no] = np.sum(((dataij*modelij/SNRtot[no][pos]**2).T*tosum).T,axis=0)#*np.mean(OBS.snr)*np.shape(modelij)[0]

    print(list_ord[no])
print("Elapsed time: "+ str(time.time()-start)+" seconds")

print()



#We have gathered the correlation values in correl_boucher
# we now pllot the snr for the selected orders in lili

lili = list_ord

sel = []
weight_fin = []
k=-1
for i in lili:
    k+=1
    try:
        sel = sel +[ np.where(list_ord==i)[0][0]]

    except:
        continue

#calculate the SNR out of planet 
a = np.sum(np.sum(correl_boucher[:,:,sel],axis=3),axis=2)
mask_std = np.ones((Nkp,Nv),dtype=bool)
for i in range(Nkp):
    for j in range(Nv):
        if ((Kp[i]>100 and Kp[i]<150) and (Vsys[j]>25 and Vsys[j]<35)):
            mask_std[i,j] = False
c = a[mask_std]
snrmap = np.std((c))

#
plt.figure()

plt.contourf(Vsys,Kp,np.sum(np.sum(correl_boucher[:,:,sel],axis=3),axis=2)/snrmap,cmap="gist_heat",levels=20)
plt.colorbar(label="SNR")
plt.savefig(pipeline_rep+"Correlated.png")
