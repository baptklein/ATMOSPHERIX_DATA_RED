import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.signal
from scipy import stats
import pickle
import time
from correlate_parameters import *
import correlate_functions as corr_func

if parallel:
    from mpi4py import MPI



correl_tot = []
#we loop here over the number of observations
for nobs in range(num_obs):
    filename = file_list[nobs]
    
    #first we read the data
    list_ord,wl,data_tot,phase,window,Stdtot,SNRtot,projtot,F,Vc,berv = corr_func.load_data(filename)

    #then we create an interpolation array of speed
    Vstarmax = np.max(np.abs(np.array(Vc)-np.array(berv)))
    Vint_max = np.max(Kpmax*np.abs(np.sin(2.0*np.pi*np.array(phase))))+np.max(np.abs([Vmin,Vmax]))+Vstarmax
    Vtot = np.linspace(-1.01*Vint_max,1.01*Vint_max,int_speed*int(Vint_max))
    
    Vstar = np.array(Vc)-np.array(berv)

    #Do we calculate on the whole data set or just when transit window > to a certain value ?
    if select_phase:
        pos = np.where(np.array(window)>min_window)
        phase2 = np.array(phase)[pos]
        window2 = np.array(window)[pos]
    else:
        pos = np.where(np.array(window)>-1.0)
        phase2 = np.array(phase)[pos]
        window2 = np.array(window)[pos]
    
    
    #Do we calculate in parallel ?
    if parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if rank==0:
            start_time = time.time()

        #share the orders between processors. need nproc<=len(list_ord)
        orders_per_process = len(list_ord) // size
        start_order = rank * orders_per_process
        end_order = (rank + 1) * orders_per_process if rank < size - 1 else len(list_ord)
        orders_to_process = list_ord[start_order:end_order+1]
        print('rank = ',rank, " and orders = ",orders_to_process)
        
        #Create interpolation on the processor
        F2D = corr_func.interpolate_model_parallel(F[start_order:end_order+1], wl[start_order:end_order+1], Vtot)
        
        #Calculate the correlation
        correl_boucher_subset = corr_func.perform_correlation(orders_to_process,\
                                                   data_tot[start_order:end_order+1],\
                                                   projtot[start_order:end_order+1],\
                                                   Stdtot[start_order:end_order+1],\
                                                   SNRtot[start_order:end_order+1],\
                                                   F2D, phase2, window2, Vstar,pos)

        comm.Barrier()  # Synchronize processes
        #gather the resuls on the root process
        correl_boucher_all = comm.gather(correl_boucher_subset, root=0)
        if rank==0:
            correl_boucher =  np.zeros((Nkp,Nv,len(list_ord),len(phase2)))
            #because we parallelize on the third dimension, this trick is necessary
            for i in range(size):
                start = i * orders_per_process
                end = (i + 1) * orders_per_process if i < size - 1 else len(list_ord)
                correl_boucher[:,:,start:end+1] = correl_boucher_all[i]
            print(np.shape(correl_boucher))
            print("time elapsed:", time.time()-start_time)
            
            
            
        # On the root process, assemble the final correl_tot
        if rank == 0:
            for i in range(size):
                correl_tot.append(correl_boucher[i])
            #plot if you want it
            if plot_ccf: 
                corr_func.plot_correlation(list_ord,correl_boucher)
            #save if you want it
            if save_ccf:
                filesave = filesave_list[nobs]
                savedata = (Kp,Vsys,list_ord,correl_boucher)
                with open(filesave, 'wb') as specfile:
                    pickle.dump(savedata,specfile)
                print("DONE")
        
    #or just in sequential    
    else:      
        start_time = time.time()
        #Interpolate the models on the speed array
        F2D = corr_func.interpolate_model(F,wl,Vtot)
        
        #And now the correlation, following boucher
        correl_boucher = corr_func.perform_correlation(list_ord, data_tot, \
                                                       projtot, Stdtot, SNRtot, F2D, \
                                                       phase2, window2, Vstar,pos)
        correl_tot.append(correl_boucher)
        print("time elapsed:", time.time()-start_time)
        #plot if you want it
        if plot_ccf: 
            corr_func.plot_correlation(list_ord,correl_boucher)
        #save if you want it !
        if save_ccf:
            filesave = filesave_list[nobs]
            savedata = (Kp,Vsys,list_ord,correl_boucher)
            with open(filesave, 'wb') as specfile:
                pickle.dump(savedata,specfile)
            print("DONE")

        


        
        


