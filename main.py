
import functions as func
import importlib
import sys

# prm_name = input("Parameter file: ")
prm_name = "parameters"
prm = importlib.import_module(prm_name)
nobs = prm.num_obs

if prm.type_obs not in ["transmission","emission"]:
    print("Observation type does not exist. it must be either 'transmission' or 'emission'")
    sys.exit()

if prm.READ_DATA:
    if len(prm.dir_data) != nobs or len(prm.read_name_fin) != nobs:
        print("dir_data and read_name_fin must be of length num_obs. Exiting")
        sys.exit()   
    for i in range(nobs):
        func.read(prm_name,prm.dir_data[i],prm.read_name_fin[i]) #We read the data, folder by folder
        
if prm.REDUCE_DATA:
    if  len(prm.reduce_name_in) != nobs or len(prm.reduce_name_out) != nobs:
        print("reduce_name_in and reduce_name_out must be of length num_obs. Exiting")
        sys.exit()
    for i in range(nobs):
        func.reduce(prm_name,prm.reduce_name_in[i],prm.reduce_name_out[i])
        
if prm.CORREL_DATA:
    if  len(prm.correl_name_in) != nobs or len(prm.correl_name_out) != nobs:
        print("reduce_name_in and reduce_name_out must be of length num_obs. Exiting")
        sys.exit()
    func.correlate(prm_name)    
        # 

