#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:43:23 2024

@author: florian
"""

import numpy as np




"""4 functions to compute V, radial velocity of the planet and/or the star, with some input parameters... """
def rvs_circular(phase,Ks,wp=0.,exc=0.):

    return -Ks*np.sin(2.*np.pi*phase)


def rvs(nu,Ks,w_peri,EXC) :

    return -Ks*(np.cos(2.*np.pi*nu + w_peri*np.pi/180) + EXC*np.cos(w_peri*np.pi/180))

def rvp_circular(phase,Kp,wp=0.,exc=0.):

    return Kp*np.sin(2.*np.pi*phase)

def rvp(nu,Kp,w_peri,EXC) :

    return Kp*(np.cos(2.*np.pi*nu + w_peri*np.pi/180) + EXC*np.cos(w_peri*np.pi/180))


"""To compute the true anomaly, following the same process as the Batman module"""
def compute_true_anomaly(Porb,EXC,T_peri,T_obs):
    n = 2*np.pi/Porb #Mean motion 
    
    M = n*(T_obs - T_peri) # mean anomaly
    
    E = getE(M, EXC); # Excentric aomaly (we have to resolve informaticaly the Kepler's equation.

    f = 2.*np.arctan(np.sqrt((1.+EXC)/(1.-EXC))*np.tan(E/2.));#True anomaly -> What we want.
    return f

def getE (M,  EXC) :  #calculates the eccentric anomaly (see Seager Exoplanets book:  Murray & Correia eqn. 5 -- see section 3)

        E = np.copy(M)
        eps = 1.0e-7
        for i in range(0,len(E)) : 

	# add fmod to ensure convergence for diabolical inputs (following Eastman et al. 2013; Section 3.1)
            while(np.fmod(np.fabs(E[i] - EXC*np.sin(E[i]) - M[i]), 2.*np.pi) > eps) : 
                fe = np.fmod(E[i] - EXC*np.sin(E[i]) - M[i], 2.*np.pi)
                fs = np.fmod(1 - EXC*np.cos(E[i]), 2.*np.pi)
                E[i] = E[i] - fe/fs
        return E
