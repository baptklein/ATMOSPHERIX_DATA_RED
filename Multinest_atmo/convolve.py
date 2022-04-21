#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 14:16:03 2021

@author: florian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:31:37 2021

@author: florian

"""

#Everything is in SI, of course !
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal


def rotate(R,freq,b,superrot) :

    c0 = 299792458.0
    #Number of point for the integral.
    #factor avoids to calculate 1/(cos(arsin(-1))) which is infinity
    #Factor must be of the order of 1-1/Nz for numerical consistency
    Nz = 500
    factor = 0.995
    z = np.linspace(-b*factor,b*factor,Nz)
    dz = z[1]-z[0]


    #From the value of dz, we define a vmax such that the second
    #order term in the Doppler effect, vmax*b/c0 be negligible in
    #front of both vmax, b and dz which is our step in speed for the
    #numerical convolution. As dz << vmax and b, we just ensure that
    # vmax*b/c0/dz = 0.1
    vmax = 0.1*dz*c0/b

    #This gives us the number of point for our interpolation
    Nint = int(2*vmax/dz)

    freq_fin = []
    R_fin = []

#The loop is simple: we define a min frequency, which is at speed -vmax.
# This gives us directly f0 and the associated max frequency, at speed vmax
#We then select the radius in this range of frequency,
#convert into speed and interpolate with a step in speed which is equal to dz.
#We can then perform the convolution.
#Finally, we define the new fmin as the previous fmax (modulo the fact
#that some points are discarded in the convolution, so actually fmin_new < fmax_old
#to ensure that all frequencies are considered) and loop until the end of the file.

    freqmax = np.min(freq)*1.01
    while freqmax<np.max(freq)*0.99:
        freqmin = freqmax
        #I like to print stuff, feel free to change it
        # print(freqmax)
        f0 = freqmin/(1.0-vmax/c0)
        freqmax = np.min([f0*(1+vmax/c0),np.max(freq*0.991)]) #we cannot have a higher freq than the data

        list_freq = np.where((freq>freqmin*0.95) & (freq<freqmax*1.05))
        speed = c0*(1.0-freq[list_freq]/f0)
        int_R = interpolate.interp1d(speed,R[list_freq])

        speed_int = np.linspace(-vmax,vmax,Nint)
        R_int = int_R(speed_int)
        zz = 1./np.cos(np.arcsin(z/b))


        #If there is superrotation, both hemispheres must be separated
        #For a blueshift, you have to take smaller speeds => start with a bunch of zeros
        #corresponding to the speed of the blueshift
        #For the redshift, it is the opposite: start at higher speed, and finish with a bunch
        #of zeros to ensure that we have the same number of points

        #As we are assuming a superrotation of 30 degres of latitude, this corresponds
        #to sin(30) = 1/2 hence we divide our convolution in 3 domain: between -90 and -60 degres,
        #there is a blueshift. Between -60 and 60, nothing. Between 60 and 90, there is a redshift.

        #Careful : because of the way python does the convolution, we have to center the convolution
        #around 0, so shift it from half the size of the convolution kernel.
        #60 degrees corresponds to sin(60) ~= 0.866, hence (1-0.866)/2*Nz points so we have to
        #shift by 0.134/4*Nz points, modulo Nz/2 to center everything

        if superrot < 0.0:
            pos = 0.0
        else:
            pos = int(superrot/dz)

        angle=60
        sinangle = np.sin(angle*np.pi/180)
        pos2 = int((1.0-sinangle)/4*Nz)
        Rconv1 = signal.oaconvolve(np.append(R_int[pos+(int(Nz/2)-pos2):],np.zeros(pos+(int(Nz/2)-pos2))), zz[:2*pos2],mode="same")
        # Rconv1 = 0.0*np.ones_like(Rconv1)
        Rconv2 = signal.oaconvolve(R_int,zz[2*pos2:-2*pos2],mode="same")
        # Rconv2 = 0.0*np.ones_like(Rconv1)
        # Rconv2 = signal.oaconvolve(np.append(np.zeros(pos),R_int[:-pos]**2), zz[:int(Nz/2)],mode="same")
        Rconv3 = signal.oaconvolve(np.append(np.zeros(pos+(int(Nz/2)-pos2)),R_int[:-(pos+(int(Nz/2)-pos2))]), zz[-2*pos2:],mode="same")
        # Rconv3 = 0.0*np.ones_like(Rconv1)
        #Don't forget the Riemann sum
        Rconv = (1.0-sinangle)/(2*pos2)*(Rconv1+Rconv3)+sinangle*2/(Nz-4*pos2)*Rconv2
        # Rconv = np.sqrt((Rconv)/np.pi)
        Rconv = Rconv/np.pi
        limits = pos+(int(Nz)-pos2)


    #######################################################################
                        ## WORD OF CAUTION ##

    #We define a list of 2 dimensions (by appending), because it is much much faster
    #than concatenating on a big list of 1 dimension for memory issue.
    #If you change that and try to have directly a 1 dimensional output (which is actually
    #performed in the last 5 lines in a few milliseconds), you will kill the speed of the program
    #######################################################################

        #At the limits, the calculation is not correct, so we exclude the border
        #which leads to redefining fmax underneath
        freq_fin.append(f0*(1.-speed_int[limits:-limits][::-1]/c0))
        R_fin.append(Rconv[limits:-limits][::-1])

         #We lost "limits" points because the convolution cannot do it on the border
         #so we redefine freqmax as freqmax minus ~Nz points to fill in the wholes
        freqmax = f0*(1.0+speed_int[int(-1.05*limits)]/c0)

    freq_fin = np.array(freq_fin)
    R_fin = np.array(R_fin)

    lentot = np.shape(freq_fin)[0]*np.shape(freq_fin)[1]
    freq_tot = freq_fin.reshape(lentot)
    R_tot = R_fin.reshape(lentot)

    #We don't send the whole model which is much too big
    diff = len(R_tot)/len(freq)
    spacing = max(1,round(diff/2))

    print(b,superrot)
    np.savetxt("/home/florian/freq2.txt",(c0/freq_tot[::spacing][::-1])*1e9)
    np.savetxt("/home/florian/Rp2.txt",R_tot[::spacing][::-1]/0.8/7e8)

    return (c0/freq_tot[::spacing][::-1])*1e9,R_tot[::spacing][::-1]

#np.savetxt("/home/florian/Bureau/Atmosphere_SPIRou/Transit_2D/Convolution/test_model/templates/allday_superrot2/lambdasallday_superrot2.txt",c0/freq_tot[::100][::-1]*1e9)
#np.savetxt("/home/florian/Bureau/Atmosphere_SPIRou/Transit_2D/Convolution/test_model/templates/allday_superrot2/Rpallday_superrot2.txt",R_tot[::100][::-1])







