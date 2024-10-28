import numpy as np
import matplotlib.pyplot as plt
import pickle
import read_functions as read_func
import reduce_functions as red_func
import time

import parameters as prm


def read(dir_data_i,name_fin_i,figure_name="transit_info.pdf"):
    ### READ FITS FILES
    print("\nRead data from",dir_data_i)
    list_ord = []
    nord     = len(prm.orders)
    for kk in range(nord):
        list_ord.append(read_func.Order(prm.orders[kk])) ### Initialize list of Order objects
    list_ord,airmass,T_obs,berv,snr_mat = read_func.read_data_spirou(prm.dir_data,list_ord,nord)
    nobs = len(T_obs)
    print("DONE")
    
    
    ### Pre-process data: correct for blaze and remove NaNs
    print("\nRemove NaNs")
    cmp = 0
    for mm in range(nord):
        O   = list_ord[cmp]   
        err = O.remove_nan()
        if err > 0: ### If only NaNs
            print("Order",O.number,"empty - removed")
            del prm.orders[cmp]
            del list_ord[cmp]
        else: cmp += 1
    nord = len(list_ord)
    print("DONE")
    
    
    print("\nCompute transit")
    ### Compute phase
    phase  = (T_obs - prm.T0)/prm.Porb
    phase -= int(phase[-1])  
    
    
    ### Compute transit window
    flux     = read_func.compute_transit(prm.Rp,prm.Rs,prm.ip,prm.T0,prm.ap,prm.Porb,prm.ep,prm.wp,prm.ld_mod,prm.ld_coef,T_obs)
    window       = (1-flux)/np.max(1-flux)
    print("DONE")
    
    ### Compute Planet-induced RV
    Vp           = read_func.get_rvs(T_obs,prm.Ks,prm.Porb,prm.T0)
    Vc           = prm.V0 + Vp - berv  #Geocentric-to-barycentric correction
    
    
    
    ### Plot transit information
    if prm.plot_read:
        print("\nPlot transit")
        TT     = 24.*(T_obs - prm.T0)
        ypad   = 15  # pad of the y label
        plt.figure(figsize=(15,12))
        # Transit flux
        ax  = plt.subplot(411)
        ax.plot(TT,flux,"-+r",label="planet")
        plt.legend(loc=3,fontsize=16)
        ax.set_ylabel("Transit curve\n", labelpad=ypad)
        # Airmass
        ax = plt.subplot(412)
        plt.plot(TT,airmass,"-k")
        ax.set_ylabel("Airmass\n", labelpad=ypad)
        # RV correction between Geocentric frame and stellar rest frame
        ax = plt.subplot(413)
        plt.plot(TT,Vc,"-k")
        ax.set_ylabel("RV correction\n[km/s]", labelpad=ypad)
        # Maximum S/N
        ax = plt.subplot(414)
        plt.plot(TT,np.max(snr_mat,axis=1),"+k")
        plt.axhline(np.mean(np.max(snr_mat,axis=1)),ls="--",color="gray")
        plt.xlabel("Time wrt transit [h]")
        ax.set_ylabel("Peak S/N\n", labelpad=ypad)
        plt.subplots_adjust(hspace=0.02)
        plt.savefig(prm.dir_figures+figure_name+".pdf",bbox_inches="tight")
        plt.close()
        print("DONE")
    
    
    ### Save as pickle
    print("\nData saved in",dir_data_i+name_fin_i)
    Ir  = []
    Ia  = []
    Bl  = []
    Ip1 = []
    Ip2 = []
    WW  = []
    SN  = []
    for nn in range(nord):
        O  = list_ord[nn]
        WW.append(O.W_raw)
        Ir.append(O.I_raw)
        Ia.append(O.I_atm)
        Bl.append(O.B_raw)
        SN.append(O.SNR)
        
    ### Namely: 
    #           - orders: List of orders -- absolute nbs
    #           - WW:     Wavelength vector for each order [list obj (N_order,N_wav)]
    #           - Ir:     Intensity values for each order [list of 2D arrays (N_order,N_obs,N_wav)]
    #           - Blaze:  Blaze values for each order [list of 2D arrays (N_order,N_obs,N_wav)]
    #           - Ia:     Telluric spectra for each order [list of 2D arrays (N_order,N_obs,N_wav)]
    #           - T_obs:  Observation dates [BJD]
    #           - phase:  Planet orbital phase - centered on mid-transit
    #           - window: Transit window (1 --> mid-transit; 0 --> out-of-transit)
    #           - berv:   BERV values [km/s]
    #           - V0+Vp:  Stellar RV [km/s]
    #           - airmass:Airmass values
    #           - SN:     Signal-to-noise values for each order [N_order,N_obs]
    
    
    savedata = (prm.orders,WW,Ir,Bl,Ia,T_obs,phase,window,berv,prm.V0+Vp,airmass,SN)
    with open(prm.dir_save_read+name_fin_i, 'wb') as specfile:
        pickle.dump(savedata,specfile)
    print("DONE")
    
    with open(prm.dir_save_read+name_fin_i[:-3]+"params") as paramfile:
        write_params = "T0 = "+str(prm.T0) +"\n"
        write_params += "Porb = "+str(prm.Porb)+"\n"
        write_params+= "Rp = "+str(prm.Rp)+"\n"
        write_params+= "Rs = "+str(prm.Rs)+"\n"
        write_params+= "ip = "+str(prm.ip)+"\n"
        write_params+= "ap = "+str(prm.ap)+"\n"
        write_params+= "ep = "+str(prm.ep)+"\n"
        write_params+= "wp = "+str(prm.wp)+"\n"
        write_params+= "ld_mod = "+prm.ld_mod+"\n"
        write_params+= "ld_coef = "+str(prm.ld_coef)+"\n"
        write_params+= "Ks = "+str(prm.Ks)+"\n"
        write_params+= "V0 = "+str(prm.V0)+"\n"
        write_params+= "inj_planet = "+str(prm.inj_planet)+"\n"
        write_params+= "amp_inj = "+str(prm.amp_inj)+"\n"

        paramfile.write(write_params)
    
def reduce(dir_reduce_i,name_in,name_out):
    T_obs,phase,window,berv,vstar,airmass,SN,list_ord= red_func.read_data_and_create_list(prm.filename)
    #Create a list of orders to remove
    nord = len(list_ord)
    ind_rem = []
    #Do we exclude some orders ?
    if len(prm.orders_rem)>0:
        for nn in range(len(list_ord)):
            if list_ord[nn].number in prm.orders_rem:
                ind_rem.append(nn)
    V_corr      = vstar - berv                 ### Geo-to-bary correction


     ### Get transits start and end indices, either prescribed or calculated
    if not prm.set_window:
        n_ini,n_end = red_func.get_transit_dates(window)    
    else:
        n_ini,n_end = prm.n_ini_fix,prm.n_end_fix


    t0          = time.time()
    NCF         = np.zeros(nord)
    file        = open(nam_info,"w")


    #### Main reduction
    print("START DATA REDUCTION")
    for nn in range(nord):
        O         = list_ord[nn]
        
        if  nn in ind_rem:
            continue
        
        print("ORDER",O.number)
        #Start by Boucher+21 telluric correction, + remove some extreme points
        O.W_cl,O.I_cl,O.A_cl,O.V_cl = red_func.tellurics_and_borders(O,dep_min,thres_up,N_bor)
        
        #if not enough points, we discard
        if len(O.W_cl) < Npt_lim:
            print("ORDER",O.number,"(",O.W_mean,"nm) discarded (",len(O.W_cl)," pts remaining)")
            print("DISCARDED\n")
            ind_rem.append(nn)
            continue

        
        #Do we include stellar correction a la Brogi ?
        if corr_star:
            O.I_cl  = red_func.stellar_from_file(O,nn,IC_name,WC_name,V_corr,sig_g,thres_up)
            
        #Delete master out of trnasit spectrum, in stellar and telluric fram
        O.W_sub, O.I_sub = O.master_out(V_corr,n_ini,n_end,sig_g,N_bor)
        
        #high pass filter: suppress modal noise
        O.W_norm1,O.I_norm1 = O.normalize(O.W_sub,O.I_sub,N_med,sig_out,N_adj,N_bor)
        ### Correct for bad pixels
        O.W_norm2,O.I_norm2= O.filter_pixel(O.W_norm1,O.I_norm1,deg_px,sig_out)

        if len(O.W_norm2) < Npt_lim:
            print("ORDER",O.number,"(",O.W_mean,"nm) discarded (",len(O.W_norm2)," pts remaining)")
            print("DISCARDED\n")
            ind_rem.append(nn)
            continue
        else:
            print(len(O.W_raw)-len(O.W_norm2),"pts removed from order",O.number,"(",O.W_mean,"nm) -- OK")
            

        #DO we detrend airmass ? 
        if det_airmass:
            O.I_fin = red_func.airmass_correction(O,airmass,deg_airmass)
        else:
            O.I_fin= O.I_norm2
        O.W_fin  = O.W_norm2
            
            
        #IF we have some weird orders, we rather delete them
        XX    = np.where(np.isnan(np.log(O.I_fin)))[0]
        if len(XX) > 0:
            print("ORDER",O.number,"intractable: DISCARDED\n")
            ind_rem.append(nn)
            continue
        
        #PCA commands
        if mode_pca =="pca":
            if auto_tune:
                O.n_com = O.tune_pca(mode_norm_pca,factor_pca,Nmap=5)
            else:
                O.n_com = npca[nn]  
            O.I_pca, O.proj = red_func.apply_PCA(O,mode_norm_pca,wpca)
        
        #Auto encoder
        elif mode_pca == "autoencoder":
           O.I_pca = encoder.apply_encoder(O.I_fin) 
           O.proj  = np.zeros((len(ff),len(ff)))
           
        #We do nothing more
        else:
            O.I_pca = O.I_fin
            O.proj  = np.zeros((len(ff),len(ff)))
        
        #Calculate the std
        red_func.calculate_final_metrics(O,N_px,file)
            

        #we can plot stuffs
        if plot_red == True and O.number == numb:
            print("Plot data reduction steps")
            lab = ["Blaze-corrected spectra","Median-corrected spectra","Normalised spectra","PCA-corrected spectra"]
            plots.plot_reduction(phase,O.W_cl,O.I_cl-O.I_cl.mean(),O.W_sub,O.I_sub-1.,O.W_fin,O.I_fin,O.W_fin,O.I_pca,lab,nam_fig)        

    file.close()        

    list_ord_fin =  np.delete(list_ord,ind_rem)
    SN_fin = np.delete(SN,ind_rem,axis=0)


    if plot_red == True:
        print("PLOT METRICS")
        plots.plot_spectrum_dispersion(list_ord_fin,nam_fig)

        

    ### Save data for correlation
    print("\nData saved in",nam_fin)
    orders_fin = []
    Iend  = []
    WW  = []
    projtot = []
    for nn in range(len(list_ord)):
        if nn in ind_rem:
            continue
        else:
            O  = list_ord[nn]
            WW.append(O.W_fin)
            Iend.append(O.I_pca)
            projtot.append(O.proj)
            orders_fin.append(O.number)
    orders_fin = np.array(orders_fin)
    savedata = (orders_fin,WW,Iend,T_obs,phase,window,berv,vstar,SN_fin,projtot)
    with open(nam_fin, 'wb') as specfile:
        pickle.dump(savedata,specfile)
    print("DONE")

    t1          = time.time()
    print("DURATION:",(t1-t0)/60.,"min")
