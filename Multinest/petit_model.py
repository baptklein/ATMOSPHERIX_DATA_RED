import numpy as np
import sys 
import prepare_model as prep_mod
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_guillot_global


class Model(object):
    def __init__(self,config_dict):

        self.p_minbar = config_dict["p_minbar"]
        self.p_maxbar= config_dict["p_maxbar"]
        self.n_pressure=config_dict["n_pressure"]
        self.P0=config_dict["P0_bar"]
        self.radius=config_dict["radius_RJ"]*cst.r_jup_mean
        self.Rs=config_dict["Rs_Rsun"]*6.9634e8
        self.gravity=config_dict["gravity_SI"]
        self.HHe_ratio=config_dict["HHe_ratio"]

        self.lambdas = config_dict["lambdas"]
        self.orderstot = config_dict["orderstot"]


        self.kappa_IR = config_dict["kappa_IR"]
        self.gamma = config_dict["gamma"]
        self.T_int = config_dict["T_int"]
        self.num_transit= config_dict["num_transit"]
        self.winds = config_dict["winds"]
        self.atmospheres= []
        self.pressures=np.logspace(self.p_minbar,self.p_maxbar,self.n_pressure)

        for i in self.orderstot: 
            atmosphere = Radtrans(self.pressures,
                                  line_species = ['1H2-16O'],
                                  rayleigh_species = ['H2', 'He'],
                                  gas_continuum_contributors=['H2-H2', 'H2-He'],
                                  wavelength_boundaries=[self.lambdas[i][0]/1000.0*0.995,self.lambdas[i][1]/1000.0*1.005],
                                  line_opacity_mode='lbl',
                                  line_by_line_opacity_sampling=4,)
            self.atmospheres.append(atmosphere)

        self.abundances = {}


#### LRS if there is any need
#        self.atmosphere_LRS = Radtrans(pressures=self.pressures,
#                                       line_species=[ 'H2O', 'CO-NatAbund', 'CO2'],
#                                       rayleigh_species=['H2', 'He'],
#                                       gas_continuum_contributors=['H2-H2', 'H2-He'],
#                                       wavelength_boundaries=[2.5, 5.5])
#




    def compute_petit(self, para_dic): # creates an atmospheric model

        temperature = para_dic["T_eq"]*np.ones_like(self.pressures)

        # Guillot temperature profile 
        
        # temperatures = temperature_profile_function_guillot_global(
        #     pressures=self.pressures,
        #     infrared_mean_opacity=self.kappa_IR,
        #     gamma=self.gamma,
        #     gravities=self.gravity_SI*100,
        #     intrinsic_temperature=self.T_int,
        #     equilibrium_temperature=para_dic["T_eq"]
        # )


        Z= 10.0**para_dic["MMR_H2O"]

        MMR_H2 = (1.0-Z)*(1-self.HHe_ratio)

        MMR_He = self.HHe_ratio*(1.0-Z)


        self.mass_fractions = {'H2':MMR_H2* np.ones_like(temperature),
                               'He': MMR_He * np.ones_like(temperature),
                               '1H2-16O': 10.0**para_dic["MMR_H2O"] * np.ones_like(temperature),
                               '12C-16O': 10.0**para_dic["MMR_CO"] * np.ones_like(temperature),
                               '12C-16O2': 10.0**para_dic["MMR_CO2"] * np.ones_like(temperature)}

#### LRS if there is any need
        # self.mass_fractions_LRS = {'H2':MMR_H2* np.ones_like(temperature),
        #                        'He': MMR_He * np.ones_like(temperature),
        #                        'H2O': 10.0**para_dic["MMR_H2O"] * np.ones_like(temperature),
        #                        'CO-NatAbund': 10.0**para_dic["MMR_CO"] * np.ones_like(temperature),
        #                        'CO2': 10.0**para_dic["MMR_CO2"] * np.ones_like(temperature)}






        MMW = 1.0/(MMR_H2/2.0+MMR_He/4.0+10.0**para_dic["MMR_H2O"]/18.0)*np.ones_like(temperature)

        wavelength_nm = []
        radius_transm = []
        #Calculate the radius order by order
        for atmo in self.atmospheres:
            wavelengths, transit_radii, _ = atmo.calculate_transit_radii(temperatures=temperature,
                                                                               mass_fractions=self.mass_fractions,
                                                                               mean_molar_masses=MMW,
                                                                               reference_gravity=self.gravity*100*para_dic["dg"],
                                                                               planet_radius=self.radius,
                                                                               reference_pressure=self.P0)

            #save it in nm and meters
            wavelength_nm.append(wavelengths*1.0e7)
            radius_transm.append(transit_radii/100.0)

        if self.winds:
            self.superrot = 0.0
            self.rot_speed = para_dic["vrot"]


#### LRS if there is any need
        # wavelengths_LRS, transit_radii_LRS, _ = self.atmosphere_LRS.calculate_transit_radii(
        # temperatures=temperature,
        # mass_fractions=self.mass_fractions_LRS,
        # mean_molar_masses=MMW,
        # reference_gravity=self.gravity*100*para_dic["dg"],
        # planet_radius=self.radius,
        # reference_pressure=self.P0)



        # wavelength_LR_microns = wavelengths_LRS*1.e4
        # radius_LR = transit_radii_LRS/100.0
        # tdepth_LR = (radius_LR/self.Rs)**2*100.
        
        

        return {
            "wavelength_nm": wavelength_nm,
            "radius_transm": radius_transm,

#### LRS if there is any need
            # "wavelength_LR_microns": wavelength_LR_microns,
            # "tdepth_LR": tdepth_LR,
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


