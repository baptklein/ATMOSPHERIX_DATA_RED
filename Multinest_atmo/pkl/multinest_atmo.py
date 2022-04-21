#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:03:12 2022

@author: florian
"""

import json
import sys
import scipy.stats, scipy
import pymultinest
from argparse import ArgumentParser
# from mpi4py import MPI
# import dill
import runpy
import os
import numpy as np
import petit_model as model
import read_data as read_d
import model_interpolate as interpmod
import likelihood_multinest as like

# MPI.pickle.__init__(dill.dumps, dill.loads)

# def mpi_print(*args):
#     if MPI.COMM_WORLD.Get_rank() == 0:
#         print(*args)
        
def mpi_print(*args):
    print(*args)
        
os.environ["OMP_NUM_THREADS"] = "1"

# First read the command line
def parse_cmdline_args():
    parser = ArgumentParser()
    g = parser.add_argument_group("Physical model")
    g.add_argument("--winds", action="store_true", help="Include rotation and or superrotation")

    g = parser.add_argument_group("Statistical model")
    g.add_argument(
        "--data",
        required=True,
        metavar="PY_FILE",
        help="Python file defining planet data (J* observations, ...) to use",
    )

    g.add_argument(
        "--like",
        choices=["Brogi","Gibson"],
        required=True,
        help="likelihood to use",
    )
    g = parser.add_argument_group("MCMC")
    g.add_argument(
        "--resume", action="store_true", help="resume from existing MCMC samples output"
    )

    args = parser.parse_args()
    mpi_print(args)
    return args

global args
args = parse_cmdline_args()


planet_data = runpy.run_path(args.data)["make_data"](args)


config = dict(
    p_minbar = -8.0,
    p_maxbar = 2.0,
    n_pressure = 130,
    mass_MJ= planet_data["mass_MJ"],
    radius_RJ=planet_data["radius_RJ"],
    Rs_Rsun = planet_data["Rs_Rsun"],
    gravity_SI= planet_data["gravity_SI"],
    P0_bar = 0.1,
    HHe_ratio=0.275,  # solar ratio
   #the limit of the SPIRou orders
    lambdas = planet_data["lambdas"],
    orderstot = planet_data["orderstot"],
    pkl = planet_data["pkl"],
    #file with the reduced  datas
    num_transit = planet_data["num_transit"],
    #finally, the eventual presence of winds
    winds = args.winds,
)

unprior = dict( kappa_IR=0.01,
        gamma=0.012,
        T_int=200,
        #T_eq=800,
        #MMR_H2O=0.001,
        MMR_CO=0.0,
        MMR_CO2=0.0,)

parameters = ["Kp","Vsys","MMR_Fe","T_eq"]
n_params = len(parameters)


config.update(unprior)


data_dic = read_d.return_data(config)
theor_spectra = model.Model(config)




def prior(cube, ndim, nparams):
    cube[0] = cube[0]*300            # uniform Kp 0:300
    cube[1] = cube[1]*80-40 # Uniform Vsys -40:40
    cube[2] = (cube[2]*7 - 8) # log-uniform prior between 10^-8 and 10^-1
    cube[3] = (cube[3]*4000 +200) # uniform between 200 and 4200


def loglike(cube, ndim, nparams):
    Kp, Vsys, H2O,T_eq= cube[0], cube[1], cube[2],cube[3]
    param_dic = dict(Kp=Kp, Vsys=Vsys,MMR_H2O=H2O,T_eq=T_eq)

    model_dic = theor_spectra.return_reduced_model(param_dic)
    model_interpolated_dic = interpmod.prepare_to_likelihood(config,model_dic,data_dic,param_dic)
    
    
    loglikelihood = like.calc_likelihood( model_interpolated_dic, args.like)
    return loglikelihood


pymultinest.run(loglike, prior, n_params,outputfiles_basename = 
                "/home/florian/Bureau/Atmosphere_SPIRou/Multinest_atmo/pkl/output_test/", 
                resume = False, verbose = True,
                n_live_points=8,
                n_iter_before_update=1)















