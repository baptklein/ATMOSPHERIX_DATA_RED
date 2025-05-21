import numpy as np
import matplotlib.pyplot as plt
import pickle
import speed_functions as speed_func
import read_functions as read_func
import reduce_functions as red_func
import correlate_functions as corr_func

import time
import plots
import importlib

from global_parameters import c0,h_planck,k_boltzmann
# import reduce_encoder as encoder


def B(lambdas,T): #Planck function as a function of wavelength for emission spectroscopy

    return 2*h_planck*c0**2/(lambdas**5)/(np.exp(h_planck*c0/lambdas/T/k_boltzmann)-1.)

def read(prm_name,dir_data_i,name_fin_i,figure_name="transit_info"):
    prm=importlib.import_module(prm_name) #import the good parameter file input from main.py
    
    ### READ FITS FILES
    print("\nRead data from",dir_data_i)
    list_ord = []
    nord     = len(prm.orders)
    for kk in range(nord):
        list_ord.append(read_func.Order(prm.orders[kk])) ### Initialize list of Order objects
    list_ord,airmass,T_obs,berv,snr_mat = read_func.read_data_spirou(dir_data_i,list_ord,nord)
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
    
    
    print("\nCompute window function and planet-induced RV")

    
    ### Compute phase and transit window. In the emission case, the phase
    ### is the true anomaly, defined from the periastron passage time. If circular,
    ### the peri astron time is equal to the conjonction time.
    if prm.type_obs=="transmission":
        phase  = (T_obs - prm.T0)/prm.Porb
        phase -= int(phase[-1])  
        flux     = read_func.compute_transit(prm.Rp,prm.Rs,prm.ip,prm.T0,prm.ap,prm.Porb,prm.ep,prm.wp,prm.ld_mod,prm.ld_coef,T_obs)
        window       = (1-flux)/np.max(1-flux)
    else:
        phase = speed_func.compute_true_anomaly(prm.Porb,prm.ep,prm.T_peri,T_obs)/2./np.pi
        if prm.transiting:
            flux     = read_func.compute_transit(prm.Rp,prm.Rs,prm.ip,prm.T0,prm.ap,prm.Porb,prm.ep,prm.wp,#
                                                     prm.ld_mod,prm.ld_coef,T_obs,ttype="secondary",T_eclipse=prm.T_eclipse,
                                                 fp = -0.1) #jsut a trick to ease the window
            window       = (1-flux)/np.max(1-flux)
        else:
            window  = np.ones(len(T_obs))
            flux  = np.ones(len(T_obs))

    print("DONE")
    
    ### Compute Planet-induced RV
    if prm.ep <1e-3:
        Vstar_planet           = speed_func.rvs_circular(phase,prm.Ks)
    else:
        Vstar_planet           = speed_func.rvs(phase,prm.Ks,prm.wp,prm.ep)
    Vc           = prm.V0 + Vstar_planet - berv  #Geocentric-to-barycentric correction
    
    
    
    ### Plot transit information
    if prm.plot_read:
        print("\nPlot observation information")
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
    
        
    #### Prepare planet injection
    if prm.INJ_PLANET:
        print("\n Inject planet")

        if prm.ep <1e-3:
            V_planet_inj = speed_func.rvp_circular(phase, prm.K_inj)+prm.V_inj
        else:
            V_planet_inj = speed_func.rvp(phase,prm.K_inj,prm.wp,prm.ep)+prm.V_inj
            
        
        if prm.type_obs=="transmission":
            W_mod,I_mod      = np.loadtxt(prm.planet_wavelength_nm_file),np.loadtxt(prm.planet_radius_m_file)
            transit_depth    = (I_mod / prm.Rs)**2
        else:
            W_mod,I_mod      = np.loadtxt(prm.planet_wavelength_nm_file),np.loadtxt(prm.planet_flux_SI_file)
            planet_flux = I_mod
    ### Save as pickle
    Ir  = []
    Ia  = []
    Bl  = []
    WW  = []
    SN  = []
    keep_ord = np.ones(nord,dtype=bool)
    
    for nn in range(nord):
        O  = list_ord[nn]
        
        ##### PLANET INJECTION #######
        if prm.INJ_PLANET:
            Wmin,Wmax  = 0.95*O.W_raw.min(),1.05*O.W_raw.max() 
            indm      = np.where((W_mod>Wmin)&(W_mod<Wmax))[0]
            W_sel     = W_mod[indm]
            if np.min(W_sel) > 0.995*np.min(O.W_raw) or np.max(W_sel) < 1.005*np.max(O.W_raw):
                print("Order",O.number,"incomplete because the injected model is not wide enough in wavelength -- discarded")
                keep_ord[nn] = 0
            else:
                if prm.type_obs=="transmission":
                    I_sel     = transit_depth[indm]
                    O.add_planet(prm.type_obs,W_sel,I_sel,window,V_planet_inj,Vc,prm.amp_inj)
                    O.I_raw = O.I_raw*(1.-O.I_syn)
                else:
                    I_sel     = planet_flux[indm]
                    O.add_planet(prm.type_obs,W_sel,I_sel,window,V_planet_inj,Vc,prm.amp_inj)
                    for tt in range(len(T_obs)):
                        ### petitRADTRANS outputs must be in SI.
                        O.I_raw[tt] = O.I_raw[tt]*(1.+O.I_syn[tt]*(prm.Rp/prm.Rs)**2/(np.pi*B((O.W_raw*1.e-9)/(1.0+(V_planet_inj[tt]+Vc[tt])/(c0/1000)),prm.T_star)))


        if keep_ord[nn]:
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
    
    savedata = (np.array(prm.orders)[keep_ord],WW,Ir,Bl,Ia,T_obs,phase,window,berv,prm.V0+Vstar_planet,airmass,SN)
    with open(prm.dir_save_read+name_fin_i, 'wb') as specfile:
        pickle.dump(savedata,specfile)
    print("\n Data saved in",prm.dir_save_read+name_fin_i)
    
    
    ##### write the parameters with the same name as the final file to keep them stored
    with open(prm.dir_save_read+name_fin_i[:-3]+"params",'w') as read_paramfile:
        write_params = "Files from" + str(dir_data_i)+"\n"
        write_params += "Observation type = "+str(prm.type_obs)+"\n"
        write_params += "T0 = "+str(prm.T0) +"\n"
        write_params += "Porb = "+str(prm.Porb)+"\n"
        write_params += "T_peri = "+str(prm.T_peri) +"\n"
        write_params += "T0_eclipse = "+str(prm.T_eclipse) +"\n"
        write_params += "Eclipsing ?  "+str(prm.transiting) +"\n"
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
        write_params+= "Add a synthetic planet = "+str(prm.INJ_PLANET)+"\n"
        if prm.INJ_PLANET:
            write_params+= "planet_wavelength_nm_file = "+str(prm.planet_wavelength_nm_file)+"\n"
            if prm.type_obs =="transmission":
                write_params+= "planet_radius_m_file = "+str(prm.planet_radius_m_file)+"\n"
            else:
                write_params+= "planet_flux_SI_file = "+str(prm.planet_flux_SI_file)+"\n"
            write_params+= "amp_inj = "+str(prm.amp_inj)+"\n"

        read_paramfile.write(write_params)
    
    
    
    
    
def reduce(prm_name,name_in,name_out):
    #Import the good parameter file as input from main.py.
    prm=importlib.import_module(prm_name)

    #Read the pkl file
    T_obs,phase,window,berv,vstar,airmass,SN,list_ord= red_func.read_data_and_create_list(prm.dir_reduce_in+name_in)
    #Create a list of orders to remove
    nord = len(list_ord)
    ind_rem = prm.ind_rem
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
    file        = open(prm.reduce_info_file,"w")

    ncomp = []

    #### Main reduction
    if prm.corr_star:
         V_brog = -berv*1000
         IS_corr = np.load(prm.IC_name)
         WS_corr = np.load(prm.WC_name)

    print("START DATA REDUCTION")
    for nn in range(nord):
        O         = list_ord[nn]
        
        if  nn in ind_rem:
            continue
        
        print("ORDER",O.number)
        #Start by Boucher+21 telluric correction, + remove some extreme points
        O.W_tells,O.I_tells,O.A_tells = red_func.tellurics_and_borders(O,prm.dep_min,prm.thres_up,prm.N_bor)

        #if not enough points, we discard
        if len(O.W_tells) < prm.Npt_lim:
            print("ORDER",O.number,"(",O.W_mean,"nm) discarded (",len(O.W_tells)," pts remaining)")
            print("DISCARDED\n")
            ind_rem.append(nn)
            continue
        
        #Remove some outliers before first normalisation
        O.W_filt,O.I_filt,ind_px= O.filter_pixel(O.W_tells,O.I_tells,prm.deg_px,prm.sig_out)

        ## Normalize before calculating master spectrum. I would advise using the percentile normalisation here
        if prm.first_norm_type =="simple":
            O.I_norm1 = np.zeros(O.I_filt[:,int(prm.N_med/2):-int(prm.N_med/2)].shape)
            for i in range(len(O.I_norm1)):
                O.I_norm1[i] = (O.I_filt[i]-np.convolve(O.I_filt[i], np.ones(prm.N_med)/prm.N_med, mode='same'))[int(prm.N_med/2):-int(prm.N_med/2)]+1
            O.W_norm1 = O.W_filt[int(prm.N_med/2):-int(prm.N_med/2)]
            
        
        elif prm.first_norm_type == "old":
            O.W_norm1,O.I_norm1 = O.normalize(O.W_filt,O.I_filt,prm.N_med,prm.sig_out,prm.N_adj,prm.N_bor)
        elif prm.first_norm_type == "percentile":
            O.I_norm1 = (O.I_filt.T/np.percentile(O.I_filt,98,axis=1)).T
            O.W_norm1 = O.W_filt
        elif prm.first_norm_type == "none":
            O.I_norm1 = O.I_filt
            O.W_norm1 = O.W_filt
            

    
        # Delete some outliers 
        O.W_cl,O.I_cl,ind_px= O.filter_pixel(O.W_norm1,O.I_norm1,prm.deg_px,prm.sig_out)
        O.V_cl = c0*(O.W_cl/O.W_mean-1.)/1000. 
        
        
        if prm.corr_star:
            ### Correction of stellar contamination (RM and center-to-limb variations)
            VS_corr = c0*(WS_corr/O.W_mean-1.)

            if (WS_corr.min()>=O.W_cl.max()) or (WS_corr.max()<=O.W_cl.min()):
                print("No stellar correction available for order:",nn)
            else:
                cov = 100.*(WS_corr.max() - WS_corr.min())/(O.W_cl.max()-O.W_cl.min())
                print("STELLAR CORRECTION: [",WS_corr.min(),",",WS_corr.max(),"]")
                print("Order wavelengths: [",O.W_cl.min(),",",O.W_cl.max(),"] - Coverage:",round(cov,1),"%")
                O.I_cl = O.correct_star(VS_corr,IS_corr,V_brog,sig_g=prm.sig_g)
                
                
        # #Delete master out of transit spectrum, in stellar and telluric fram

        O.W_sub, O.I_sub = O.master_out(V_corr,n_ini,n_end,prm.sig_g,prm.N_bor)

        #second normalisation. I would recommend simple or old to delete modal noise,
        # but you can try not to do it for test
        if prm.second_norm_type =="simple":
            O.I_norm2 = np.zeros(O.I_sub[:,int(prm.N_med/2):-int(prm.N_med/2)].shape)
            for i in range(len(O.I_norm2)):
                O.I_norm2[i] = (O.I_sub[i]-np.convolve(O.I_sub[i], np.ones(prm.N_med)/prm.N_med, mode='same'))[int(prm.N_med/2):-int(prm.N_med/2)]+1
            O.W_norm2 = O.W_sub[int(prm.N_med/2):-int(prm.N_med/2)]
            
        
        elif prm.second_norm_type == "old":
            
            O.W_norm2,O.I_norm2 = O.normalize(O.W_sub,O.I_sub,prm.N_med,prm.sig_out,prm.N_adj,prm.N_bor)
            
        elif prm.second_norm_type == "percentile":
            O.I_norm2 = (O.I_sub.T/np.percentile(O.I_sub,98,axis=1)).T
            O.W_norm2 = O.W_sub
            
        elif prm.second_norm_type == "none":
            O.I_norm2 = O.I_sub
            O.W_norm = O.W_sub
            
        ### Correct for bad pixels
        O.W_norm2_pix,O.I_norm2_pix,tmp= O.filter_pixel(O.W_norm2,O.I_norm2,prm.deg_px,prm.sig_out)


        if len(O.W_norm2_pix) < prm.Npt_lim:
            print("ORDER",O.number,"(",O.W_mean,"nm) discarded (",len(O.W_norm2_pix)," pts remaining)")
            print("DISCARDED\n")
            ind_rem.append(nn)
            continue
        else:
            print(len(O.W_raw)-len(O.W_norm2_pix),"pts removed from order",O.number,"(",O.W_mean,"nm) -- OK")
            

        #DO we detrend airmass ? 
        if prm.det_airmass:
            I_log           = np.log(O.I_norm2_pix)
            I_det_log       = O.detrend_airmass(O.W_norm2_pix,O.I_norm2_pix,airmass,prm.deg_airmass)
            I_det           = np.exp(I_det_log)
            O.I_fin = I_det
        else:
            O.I_fin= O.I_norm2_pix
        O.W_fin  = O.W_norm2_pix
            
            
        #IF we have some weird orders, we rather delete them
        XX    = np.where(np.isnan(np.log(O.I_fin)))[0]
        if len(XX) > 0:
            print("ORDER",O.number,"intractable: DISCARDED\n")
            ind_rem.append(nn)
            continue
        
        #PCA commands
        if prm.mode_pca =="pca":
            if prm.auto_tune:
                O.n_com = O.tune_pca(prm.mode_norm_pca,prm.factor_pca,Nmap=5,min_pca=prm.min_pca)
            else:
                O.n_com = prm.npca[nn]  
            O.I_pca, O.proj = red_func.apply_PCA(O,prm.mode_norm_pca,prm.wpca)
        
        #Auto encoder. NOT USABLE RIGHT NOW
        elif prm.mode_pca == "autoencoder":
           # O.I_pca = encoder.apply_encoder(O.I_fin) 
           O.proj  = np.zeros((len(O.I_fin),len(O.I_fin)))
           
        #We do nothing more
        else:
            O.I_pca = O.I_fin
            O.proj  = np.zeros((len(O.I_fin),len(O.I_fin)))
        
        ncomp.append(O.n_com)
        
        #Calculate the std
        red_func.calculate_final_metrics(O,prm.N_px,file)
            

        #we can plot stuffs
        if prm.plot_red == True and O.number == prm.numb:
            print("Plot data reduction steps")
            figure_reduce_name =  prm.dir_figures+name_out[:-4]+"_reduction_" + str(prm.numb) + ".png"
            lab = ["Blaze-corrected spectra","Median-corrected spectra","Normalised spectra","PCA-corrected spectra"]
            plots.plot_reduction(phase,O.W_cl,O.I_cl-O.I_cl.mean(),O.W_sub,O.I_sub-1.,O.W_fin,O.I_fin-1.,O.W_fin,O.I_pca,lab,figure_reduce_name)        

    file.close()        

    list_ord_fin =  np.delete(list_ord,ind_rem)
    SN_fin = np.delete(SN,ind_rem,axis=0)


    if prm.plot_red == True:
        print("PLOT METRICS")
        figure_metrics_name=prm.dir_figures+name_out[:-4]+"_metrics_" + str(prm.numb) + ".png"
        plots.plot_spectrum_dispersion(list_ord_fin,figure_metrics_name)

        

    ### Save data for correlation
    print("\nData saved in",name_out)
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
    with open(prm.dir_reduce_out+name_out, 'wb') as specfile:
        pickle.dump(savedata,specfile)
    print("DONE")

    t1          = time.time()
    print("DURATION:",(t1-t0)/60.,"min")
    
    
    ##### write the parameters with the same name as the final file to keep them stored
    with open(prm.dir_reduce_out+name_out[:-3]+"params",'w') as reduc_paramfile:
        write_params = "Input read pkl file is "+str(prm.dir_reduce_in+name_in)+"\n"
        write_params += "Stellar model = "+str(prm.corr_star) +"\n"
        if prm.corr_star:
            write_params += "Wavelength star = "+str(prm.WC_name)+"\n"
            write_params+= "Intensity star = "+str(prm.IC_name)+"\n"
        write_params += "dep_min = "+str(prm.dep_min)+"\n"
        write_params+= "thres_up = "+str(prm.thres_up)+"\n"
        write_params+= "Npt_lim = "+str(prm.Npt_lim)+"\n"
        write_params+= "pixel = "+str(prm.pixel)+"\n"
        write_params+= "sig_g = "+str(prm.sig_g)+"\n"
        write_params+= "N_bor = "+str(prm.N_bor)+"\n"
        write_params+= "norm1 = "+str(prm.first_norm_type)+"\n"
        write_params+= "norm2 = "+str(prm.second_norm_type)+"\n"
        write_params+= "N_med = "+str(prm.N_med)+"\n"
        write_params+= "sig_out = "+str(prm.sig_out)+"\n"
        write_params+= "N_adj = "+str(prm.N_adj)+"\n"
        write_params+= "deg_px = "+str(prm.deg_px)+"\n"
        write_params+= "det_airmass = "+str(prm.det_airmass)+"\n"
        if prm.det_airmass:
            write_params+= "deg_airmass = "+str(prm.deg_airmass)+"\n"
        write_params+= "mode_pca = "+prm.mode_pca+"\n"
        write_params+= "wpca= "+str(prm.wpca)+"\n"
        write_params+= "auto_tune = "+str(prm.auto_tune)+"\n"
        if prm.auto_tune:
            write_params+= "factor_auto_tune = "+str(prm.factor_pca)+"\n"
            write_params+= "min_pca = "+str(prm.min_pca)+"\n"
        write_params+= "mode_norm_pca = "+prm.mode_norm_pca+"\n"

        write_params += "Number of components = "+str(ncomp)+"\n"
        if len(prm.orders_rem)>0:
            write_params += "orders_rem = "+str()
        write_params += "set_window = "+str(prm.set_window) +"\n"
        if prm.set_window:
            write_params += "n_ini_fix= "+str(prm.n_ini_fix) +"\n"
            write_params += "n_end_pix = "+str(prm.n_end_fix) +"\n"
        else:
            write_params += "n_ini_auto= "+str(prm.n_ini_fix) +"\n"
            write_params += "n_end_auto = "+str(prm.n_end_fix) +"\n"

        reduc_paramfile.write(write_params)





def correlate(prm_name):
    #Import the good parameter file as input from main.py.

    prm=importlib.import_module(prm_name)

    
    if prm.parallel:
        from mpi4py import MPI


    correl_tot = []
    list_ord_tot = []

    #we loop here over the number of observations
    for nobs in range(prm.num_obs):
        filename = prm.dir_correl_in+prm.correl_name_in[nobs]        
        #first we read the data
        list_ord,wl,data_tot,phase,window,Stdtot,SNRtot,projtot,F,Vc,berv = corr_func.load_data(filename,prm.select_ord,
                                                                                                prm.list_ord_correl,prm.dir_correl_mod)
        #then we create an interpolation array of speed
        Vstarmax = np.max(np.abs(np.array(Vc)-np.array(berv)))
        
        Vint_max = np.max(np.abs(speed_func.rvp(phase,prm.Kpmax,prm.wp,prm.ep)))+np.max(np.abs([prm.Vmin,prm.Vmax]))+Vstarmax
        Vtot = np.linspace(-1.02*Vint_max,1.02*Vint_max,prm.int_speed*int(Vint_max))
        
        Vstar = np.array(Vc)-np.array(berv)
    
        #Do we calculate on the whole data set or just when transit window > to a certain value ?
        if prm.select_phase:
            pos = np.where(np.array(window)>prm.min_window)
            phase2 = np.array(phase)[pos]
            window2 = np.array(window)[pos]
        else:
            pos = np.where(np.array(window)>-1.0)
            phase2 = np.array(phase)[pos]
            window2 = np.array(window)[pos]
        
        
        #Do we calculate in parallel ?
        if prm.parallel:
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
            F2D = corr_func.interpolate_model_parallel(F[start_order:end_order+1], wl[start_order:end_order+1], Vtot,prm.pixel_correl,prm.weights)
            
            #Calculate the correlation
            if prm.ep<1e-3:
                speed_planet = speed_func.rvp_circular
            else:
                speed_planet = speed_func.rvp
            correl_boucher_subset = corr_func.perform_correlation(orders_to_process,\
                                                       data_tot[start_order:end_order+1],\
                                                       projtot[start_order:end_order+1],\
                                                       Stdtot[start_order:end_order+1],\
                                                       SNRtot[start_order:end_order+1],\
                                                       F2D, phase2, window2, Vstar,pos, \
                                                       prm.Kp_array,prm.Vsys_array,prm.nbor_correl,prm.use_proj,
                                                       prm.proj_fast,prm.mode_norm_pca_correl,speed_planet,ecc=prm.ep,wp=prm.wp)
    
    
            comm.Barrier()  # Synchronize processes
            #gather the resuls on the root process
            correl_boucher_all = comm.gather(correl_boucher_subset, root=0)
            if rank==0:
                correl_boucher =  np.zeros((prm.Nkp,prm.Nv,len(list_ord),len(phase2)))
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
                    correl_tot.append(correl_boucher)
                    list_ord_tot.append(list_ord)
                #plot if you want it
                if prm.plot_ccf_indiv: 
                    corr_func.plot_correlation(list_ord,correl_boucher,prm.select_plot,prm.list_ord_plot_correl,
                                               prm.Kp_array,prm.Vsys_array,\
                                         prm.Kp_min_std,prm.Kp_max_std,prm.Vsys_min_std,prm.Vsys_max_std,prm.nlevels,\
                                         prm.white_lines,prm.Kp_planet,prm.Vsys_planet)
            
        #or just in sequential    
        else:      
            start_time = time.time()
            #Interpolate the models on the speed array
            F2D = corr_func.interpolate_model(F,wl,Vtot,prm.pixel_correl,prm.weights)
            
            
            if prm.ep<1e-3:
                speed_planet = speed_func.rvp_circular
            else:
                speed_planet = speed_func.rvp
            
            #And now the correlation, following boucher
            correl_boucher = corr_func.perform_correlation(list_ord, data_tot, \
                                                           projtot, Stdtot, SNRtot, F2D, \
                                                           phase2, window2, Vstar,pos,\
                                                           prm.Kp_array,prm.Vsys_array,prm.nbor_correl,prm.use_proj,
                                                           prm.proj_fast,prm.mode_norm_pca_correl,speed_planet,prm.ep,prm.wp)
            correl_tot.append(correl_boucher)
            list_ord_tot.append(list_ord)
    
            print("time elapsed:", time.time()-start_time)
            #plot if you want it
            if prm.plot_ccf_indiv: 
                corr_func.plot_correlation(list_ord,correl_boucher,prm.select_plot,prm.list_ord_plot_correl,
                                           prm.Kp_array,prm.Vsys_array,\
                                     prm.Kp_min_std,prm.Kp_max_std,prm.Vsys_min_std,prm.Vsys_max_std,prm.nlevels,\
                                     prm.white_lines,prm.Kp_planet,prm.Vsys_planet)

            
        #save if you want it
        if prm.save_ccf:
            filesave = prm.dir_correl_out+prm.correl_name_out[nobs]
            savedata = (prm.Kp_array,prm.Vsys_array,list_ord,correl_boucher)
            with open(filesave, 'wb') as specfile:
                pickle.dump(savedata,specfile)
            print("DONE")
            
            ##### write the parameters with the same name as the final file to keep them stored
            with open(prm.dir_correl_out+prm.correl_name_out[nobs][:-3]+"params",'w') as correl_paramfile:
                write_params = "Reduced file : " + str(prm.dir_correl_in+prm.correl_name_in[nobs])+"\n"
                write_params += "parallel = "+str(prm.parallel) +"\n"
                write_params += "pixel_correl = "+str(prm.pixel_correl)+"\n"
                write_params+= "weights = "+str(prm.weights)+"\n"
                write_params+= "Kpmin = "+str(prm.Kpmin)+"\n"
                write_params+= "Kpmax = "+str(prm.Kpmax)+"\n"
                write_params+= "Nkp = "+str(prm.Nkp)+"\n"
                write_params+= "Vsysmin = "+str(prm.Vmin)+"\n"
                write_params+= "Vsysmax = "+str(prm.Vmax)+"\n"
                write_params+= "Nv= "+str(prm.Nv)+"\n"
                write_params+= "model from :  "+str(prm.dir_correl_mod)+"\n"
                write_params+= "orders = "+str(list_ord)+"\n"
                write_params+= "select_phase = "+str(prm.select_phase)+"\n"
                if prm.select_phase:
                    write_params+= "min_window = "+str(prm.min_window)+"\n"
                write_params+= "int_speed = "+str(prm.int_speed)+"\n"
                write_params+= "nbor_correl = "+str(prm.nbor_correl)+"\n"
                write_params+= "use_proj= "+str(prm.use_proj)+"\n"
                if prm.use_proj:
                    write_params+= "proj_fast = "+str(prm.proj_fast)+"\n"
                    write_params+= "mode_norm_pca_correl = "+str(prm.mode_norm_pca_correl)+"\n"
                if prm.plot_ccf_indiv:
                    write_params+= "Kpmin_std = "+str(prm.Kpmin)+"\n"
                    write_params+= "Kpmax_std = "+str(prm.Kpmax)+"\n"
                    write_params+= "Vsysmin_std = "+str(prm.Vmin)+"\n"
                    write_params+= "Vsysmax_std = "+str(prm.Vmax)+"\n"
                    write_params+= "nlevels = "+str(prm.nlevels)+"\n"
                    write_params+= "white_lines = "+str(prm.white_lines)+"\n"
                    write_params+= "KP_planet = "+str(prm.Kp_planet)+"\n"
                    write_params+= "Vsys_planet = "+str(prm.Vsys_planet)+"\n"

                correl_paramfile.write(write_params)
            

            
    if prm.plot_ccf_tot:
        if len(correl_tot)>1:
            if prm.parallel:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                if rank ==0:
                    corr_func.plot_correlation_tot(list_ord_tot,correl_tot,prm.select_plot,prm.list_ord_plot_correl,
                                                   prm.Kp_array,prm.Vsys_array,\
                                         prm.Kp_min_std,prm.Kp_max_std,prm.Vsys_min_std,prm.Vsys_max_std,prm.nlevels,\
                                         prm.white_lines,prm.Kp_planet,prm.Vsys_planet)
            else:
                corr_func.plot_correlation_tot(list_ord_tot,correl_tot,prm.select_plot,prm.list_ord_plot_correl,
                                               prm.Kp_array,prm.Vsys_array,\
                                     prm.Kp_min_std,prm.Kp_max_std,prm.Vsys_min_std,prm.Vsys_max_std,prm.nlevels,\
                                     prm.white_lines,prm.Kp_planet,prm.Vsys_planet)
    
            
        


