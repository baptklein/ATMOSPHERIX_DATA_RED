import numpy as np
#import lmfit
import prepare_model as prep_mod
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from scipy.integrate import simps



class Model(object):
    def __init__(self,config_dict):

        self.p_minbar = config_dict["p_minbar"]
        self.p_maxbar= config_dict["p_maxbar"]
        self.n_pressure=config_dict["n_pressure"]
        self.P0=config_dict["P0_bar"]
        self.radius=config_dict["radius_RJ"]*nc.r_jup_mean
        self.Rs=config_dict["Rs_Rsun"]*7.0e8
        self.gravity=config_dict["gravity_SI"]
        self.HHe_ratio=config_dict["HHe_ratio"]

		#now the transit
        self.inc = config_dict["inc"]
        self.t0 = config_dict["t0"]
        self.sma = config_dict["sma"]
        self.orb_per = config_dict["orb_per"]
        self.ecc =config_dict["ecc"]
        self.w_peri = config_dict["w_peri"]
        self.limbdark = config_dict["limbdark"]
        self.u_limbdark = config_dict["u_limbdark"]
        self.dates = config_dict["dates"]

        self.Wmean = config_dict["Wmean"]
        self.lambdas = config_dict["lambdas"]
        self.orderstot = config_dict["orderstot"]
        self.orders = config_dict["orders"]

        self.Vfiles =config_dict["Vfiles"]
        self.Ifiles =config_dict["Ifiles"]
        self.Stdfiles =config_dict["Stdfiles"]

        if not self.Stdfiles:
            self.Stdfiles = config_dict["Ifiles"]

        self.kappa_IR = config_dict["kappa_IR"]
        self.gamma = config_dict["gamma"]
        self.T_int = config_dict["T_int"]
        #self.T_eq = config_dict["T_eq"]
        self.num_transit= config_dict["num_transit"]
        # self.MMR_H2O = config_dict["MMR_H2O"]
        # self.MMR_CO2 = config_dict["MMR_CO2"]
        # self.MMR_CO = config_dict["MMR_CO"]

        self.winds = config_dict["winds"]
#         self.atmosphere= Radtrans(line_species = ['H2O_main_iso','CO2_main_iso','CO_main_iso'], \
# 		  rayleigh_species = ['H2', 'He'], \
# 		  continuum_opacities = ['H2-H2'], \
# 		  wlen_bords_micron = [self.wlen_min,self.wlen_max], \
# 		  mode = 'lbl')

        self.atmospheres= []
        self.pressures=np.logspace(self.p_minbar,self.p_maxbar,self.n_pressure)

        for i in self.orderstot: 
            atmosphere = Radtrans(line_species = ['H2O_main_iso'], \
		  rayleigh_species = ['H2', 'He'], \
		  continuum_opacities = ['H2-H2'], \
		  wlen_bords_micron = [self.lambdas[i][0]/1000.0*0.985,self.lambdas[i][1]/1000.0*1.015], \
                  lbl_opacity_sampling = 4, \
		  mode = 'lbl')
            atmosphere.setup_opa_structure(self.pressures)

            self.atmospheres.append(atmosphere)

        self.abundances = {}


    def compute_petit(self, para_dic): # creates an atmospheric model
        # temperature=nc.guillot_global(self.pressures, 10.0**para_dic["kappa_IR"], \
# 		10.0**para_dic["gamma"], self.gravity, para_dic["T_int"], para_dic["T_eq"])
#         temperature=nc.guillot_global(self.pressures, self.kappa_IR, \
# 		self.gamma, self.gravity*100.0, self.T_int, self.T_eq)
        temperature = para_dic["T_eq"]*np.ones_like(self.pressures)

        Z= 10.0**para_dic["MMR_H2O"]#+10.0**para_dic["MMR_CO"]
        # Z= self.MMR_H2O+self.MMR_CO2+self.MMR_CO


        MMR_H2 = (1.0-Z)*(1-self.HHe_ratio)

        MMR_He = self.HHe_ratio*(1.0-Z)


        self.abundances = {}
        self.abundances['H2'] = MMR_H2* np.ones_like(temperature)

        self.abundances['He'] = MMR_He * np.ones_like(temperature)

        self.abundances['H2O_main_iso'] = 10.0**para_dic["MMR_H2O"] * np.ones_like(temperature)
        # self.abundances['CO2_main_iso'] = 10.0**para_dic["MMR_CO2"] * np.ones_like(temperature)
        #self.abundances['CO_main_iso'] = 10.0**para_dic["MMR_CO"] * np.ones_like(temperature)

        # self.abundances['H2O_main_iso'] = self.MMR_H2O * np.ones_like(temperature)
        # self.abundances['CO2_main_iso'] = self.MMR_CO2 * np.ones_like(temperature)
        # self.abundances['CO_main_iso'] = self.MMR_CO * np.ones_like(temperature)

		#MMW = (sum(Zi/mi))**-1)


        MMW = 1.0/(MMR_H2/2.0+MMR_He/4.0+10.0**para_dic["MMR_H2O"]/18.0)*np.ones_like(temperature)
                    # 10.0**para_dic["MMR_CO2"]/48.0 +
                    #                   +10.0**para_dic["MMR_CO"]/28.0
                    # )


        # MMW = 1.0/(MMR_H2/2.0+MMR_He/4.0+self.MMR_H2O/18.0+
                   # self.MMR_CO2/48.0 +
                   # self.MMR_CO/28.0)*np.ones_like(temperature)
        freq_nm = []
        radius_transm = []
        for atmo in self.atmospheres:
            atmo.calc_transm(temperature, self.abundances, self.gravity*100, MMW, R_pl=self.radius, P0_bar=self.P0)

            freq_nm.append(nc.c/atmo.freq/1.0e-7)
            radius_transm.append(atmo.transm_rad/100.0)

        if self.winds:
            self.superrot = para_dic["superrot"]
            self.rot_speed = para_dic["rot_speed"]


        return {
            "freq_nm": freq_nm,
            "radius_transm": radius_transm,
        }

    def reduce_model(self, model_dic): #renormalizes the atmospheric mode
        #print(self.winds)
        if self.winds:
            return prep_mod.prepare(model_dic,self.Rs,self.orderstot,winds=self.winds,rot_speed=
                              self.rot_speed,superrot=self.superrot)
        else:
            return prep_mod.prepare(model_dic,self.Rs,self.orderstot)


    def return_reduced_model(self,para_dic):
        model_dic = self.compute_petit(para_dic)
        return self.reduce_model(model_dic)


