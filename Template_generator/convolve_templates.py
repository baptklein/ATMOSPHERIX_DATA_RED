"""
Created on Fri Aug 13 13:31:37 2021

@author: florian

"""

#Everything is in SI, of course !
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal
import time


def rotate(R,wl,vrot,superrot,angle_super=25.0*np.pi/180,sigma=1270.0) :

    if vrot<100: #we are not able to see the difference anyway , and that prevents from creating exceptions in 
    #the code
        vrot = 100
    
    c0 = 299792458.0

    #we take a kernel of size 50 000 km/s, it is exagerated but is safe

    vlim = 50000.0
    nv = 5000
    v = np.linspace(-vlim,vlim,nv)
    dv = v[1]-v[0]
    #we interpolate the model onto a regularly spaced speed array
    w0 = np.mean(wl)
    speed = c0*(w0/wl-1)
    speed_int = np.arange(0.995*np.min(speed),0.995*np.max(speed),step=dv)
    fmod = interpolate.interp1d(speed,R)
    mod_int = fmod(speed_int)

    
    #we prepare convolution by defining an appropriate speed array
    n = int(2*vrot/dv) 
    fraclim = dv*n/2/vrot
    vrot_array = np.arange(-vrot*fraclim,vrot,step=dv) #a trick to ensure symmetry in this array
    first_conv=  1/np.sqrt(1-(vrot_array/vrot)**2) #rotation kernel
    second_conv = np.exp(-v**2/2/sigma**2) #instrumental kernel

    if superrot>100.0: #if lower, we won't see the difference anyway
        #limits of convolution for superrotating parts
        cos1 = np.cos(angle_super)
        pos1 = np.where(vrot_array/vrot<-cos1)[0][-1]
        first_conv_1 = np.zeros(n+1) #to ensure same size, we fill the convoluant with zeros
        first_conv_2 = np.zeros(n+1) 
        first_conv_1[:pos1+1] = 1/np.sqrt(1-((vrot_array[:pos1+1])/vrot)**2)
        cos2 = -cos1
        pos2 = np.where(vrot_array/vrot>-cos2)[0][0]
        first_conv_2[pos2:] = 1/np.sqrt(1-((vrot_array[pos2:])/vrot)**2)
        nsuper_conv = int(superrot/dv)
        if nsuper_conv%2 ==1:
            nsuper_conv+=1
        nsuper_half = int(nsuper_conv/2) #in the worst case this leads to an error of 5 m/s ... that's allright


        conv1_mod = 1/vrot/np.pi*((vrot-cos1*(vrot))/(pos1+1)*signal.oaconvolve(mod_int[2*nsuper_conv:]**2, first_conv_1,mode="same") + \
                                  (2*cos1*vrot)/(len(vrot_array)-2*pos1-2)*signal.oaconvolve(mod_int[2*nsuper_half:-2*nsuper_half]**2, first_conv[pos1+1:pos2],mode="same") + \
                                  (vrot-cos1*(vrot))/(pos1+1)*signal.oaconvolve(mod_int[:-2*nsuper_conv]**2, first_conv_2,mode="same"))
        conv2_mod = 1/sigma/np.sqrt(2*np.pi)*2*vlim/nv*signal.oaconvolve(conv1_mod,second_conv,mode="same")  
        convtot_mod = np.sqrt(conv2_mod)
        wl_int = w0/(1+speed_int[nsuper_conv:-nsuper_conv]/c0)

        
    else:
        conv1_mod = 1/vrot/np.pi*((np.max(vrot_array)-np.min(vrot_array))/len(vrot_array)*signal.oaconvolve(mod_int**2, first_conv,mode="same"))#+(0.0447251+0.051)*vrot*np.mean(mod_int**2))
        conv2_mod = 1/sigma/np.sqrt(2*np.pi)*2*vlim/nv*signal.oaconvolve(conv1_mod,second_conv,mode="same")  
        convtot_mod = np.sqrt(conv2_mod)
        wl_int = w0/(1+speed_int/c0)


    #the model is largely oversampled and we don't want that, it is going to kill the calculation time
    diff = len(convtot_mod)/len(wl)
    spacing = max(1,round(5.*diff/6))


    return (wl_int[n+nv:-n-nv:spacing],convtot_mod[n+nv:-n-nv:spacing])