import numpy as np
import pickle

def return_data(config_dict):
    

    orders_data = []
    wmean_data = []
    V_data = []
    intensity_data = []
    std_data = []
    phase_data = []
    window_data = []
    proj_data = []
    V_star = []
    for j in range(config_dict["num_transit"]):
        with open(config_dict["pkl"][j][0],'rb') as ccfile:
            orders,Wmean,V,I,Std,phase,window,Vtot,proj = pickle.load(ccfile)
        orders_data.append(orders)
        wmean_data.append(Wmean)
        V_data.append(V)# same here for the data   
        intensity_data.append(I)
        std_data.append(Std)
        phase_data.append(phase)
        window_data.append(window)
        V_star.append(Vtot)
        proj_data.append(proj)
        #print(len(proj))
        #exit()        


#
    uncertainties_LR = []
    data_LR = []
    data_LR_wavelength = []
    data_table = np.loadtxt(config_dict["LRS_file"], skiprows=1)
    for i in range(len(data_table)): #reading the data file
        data_LR_wavelength.append(data_table[i][0])
        data_LR.append(data_table[i][1])
        uncertainties_LR.append(data_table[i][2])
#

 #final_data and final std contain num_transit arrays of size norders*nphase.
 #each cnsecutive array is a new phase, until all phases are explored 
 #and you get to a new order.
 
#final V contains num_transit arrays of size norders,
#as we don't require a phase dependency.
    #print(len(proj_data))
    #exit()

    return {
			"orders": orders_data,
       "wmean" : wmean_data,  
            "V"    : V_data,
			"intensity": intensity_data,
            "std" : std_data,
            "phase" : phase_data,
            "window" : window_data,
            "Vstar" : V_star,
            "proj" : proj_data,
            "data_LR" : data_LR,
            "data_LR_wavelength" : data_LR_wavelength,
            "uncertainties_LR" :uncertainties_LR ,
        }
