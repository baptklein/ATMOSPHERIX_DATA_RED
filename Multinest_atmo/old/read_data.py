import numpy as np




def return_data(config_dict):
       
    final_data = []
    final_std = []
    final_V = []
    for j in range(config_dict["num_transit"]):
        print(j)
        transit_data = []
        transit_V = []
        transit_std = []
        for i in range(len(config_dict["orders"][j])):
            # print(config_dict[".Wmean[i])
            #print(config_dict["Vfiles"][j][i])            
            V_data =  np.loadtxt("data/"+config_dict["Vfiles"][j][i])
            I_data = np.loadtxt("data/"+config_dict["Ifiles"][j][i])
            Std_data =  np.loadtxt("data/"+config_dict["Stdfiles"][j][i])

            indiv_data = []
            indiv_std = []
            for n in range(len(I_data)):

                data_tmp =I_data[n]- np.mean(I_data[n])
                indiv_data.append(data_tmp)
                indiv_std.append(Std_data)
                
            transit_V.append(V_data[0])
            transit_data.append(indiv_data)
            transit_std.append(indiv_std)

        final_data.append(transit_data)
        final_std.append(transit_std)
        final_V.append(transit_V)# same here for the data   

 #final_data and final std contain num_transit arrays of size norders*nphase.
 #each cnsecutive array is a new phase, until all phases are explored 
 #and you get to a new order.
 
#final V contains num_transit arrays of size norders,
#as we don't require a phase dependency.

    return {
			"intensity": final_data,
            "std" : final_std,
            "V"    : final_V
        }
