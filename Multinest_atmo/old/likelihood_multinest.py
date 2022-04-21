import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
import pandas as pd


def calc_likelihood(corr,like_type):
    like = np.zeros(len(corr["data"]))
    
    
    # plt.plot((corr["model"][0][]))


    if like_type=="Brogi":
        try :
            (len(corr["data"]) == len(corr["model"]))
        except:
            raise NameError("data and model not equal")
            exit()
        for i  in range(len(corr["data"])):
            N = len(corr["data"][i])
            sf  = np.var(corr["data"][i])
            sg  = np.var(corr["model"][i])
            Rs = 1./N*np.sum(np.array(corr["data"][i])*np.array(corr["model"][i]))
            like[i]= -N/2.*np.log(sf+sg-2.*Rs)

    
           
    elif like_type=="Gibson":
        
        for i  in range(len(corr["data"])):
            N = len(corr["data"][i])
            tolog = np.sum((np.array(corr["data"][i])-np.array(corr["model"][i]))**2/np.array(corr["std"][i])**2)/N
            if tolog<=0.0:
                print(tolog)
            like[i]= -N/2.*(np.log(tolog))
        
    return np.sum(like)
