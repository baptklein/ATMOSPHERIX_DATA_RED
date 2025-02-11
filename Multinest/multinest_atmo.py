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
import likelihood_multinest as like_HR
import likelihood_LR as like_LR
from priors import Priors

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
        choices=["Brogi","Gibson","Gibson_global","Gibson_transit"],
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
    #the eventual presence of winds
    winds = args.winds,
    #name of the LRS file (e.g. HST or JWST) if any
    LRS_file = planet_data["LRS_file"],
)

unprior = dict( kappa_IR=0.001,
        gamma=10**-1.5,
        T_int=500,)

parameters = ["Kp","Vsys","H2O","T_eq"]
n_params = len(parameters)


config.update(unprior)


data_dic = read_d.return_data(config)
theor_spectra = model.Model(config)



pri=Priors()
def prior(cube, ndim, nparams):
    cube[0] = pri.UniformPrior(cube[0], 50., 200.)           # uniform Kp 0:300
    cube[1] =  pri.UniformPrior(cube[1], 0.,20.)           # un
    cube[2] =  pri.UniformPrior(cube[2], -8., -1.0)           # un
    cube[3] =  pri.UniformPrior(cube[3], 100.,3000.)            # un

def loglike(cube, ndim, nparams):
    Kp, Vsys,H2O,T_eq = cube[0], cube[1], cube[2],cube[3]
    param_dic = dict(Kp=Kp, Vsys=Vsys,MMR_H2O=H2O,T_eq=T_eq)

    model_dic = theor_spectra.return_reduced_model(param_dic)

    model_interpolated_dic = interpmod.prepare_to_likelihood(config,model_dic,data_dic,param_dic)
    
    
    loglikelihood_HR = like_HR.calc_likelihood_HR( model_interpolated_dic, args.like)

    #### LRS if there is any need
#    loglikelihood_LR = like_LR.return_like_LR( model_dic, data_dic)


    return loglikelihood_HR#+loglikelihood_LR


pymultinest.run(loglike, prior, n_params,outputfiles_basename = 
               "/home/adminloc/Bureau/Atmospheres/Pipeline_v2/ATMOSPHERIX_DATA_RED/Multinest/example/", 
                resume = False, verbose = True,
                n_live_points=400,
                n_iter_before_update=1)















