import numpy as np
import sys
#from petitRADTRANS import Radtrans
#from petitRADTRANS import nat_cst as nc
#from scipy.integrate import simps
#from petitRADTRANS.physics import guillot_global
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_guillot_global


radius_RJ = 0.8897
gravity_SI = 1.42
Rs_Rsun = 1.38
p_minbar = -8.0
p_maxbar = 2.0
n_pressure = 130
P0_bar = 0.1
HHe_ratio=0.275  # solar ratio
   #the limit of the

radius=radius_RJ*cst.r_jup_mean
Rs=Rs_Rsun*6.9634e8


post = np.loadtxt("../crires-JWST_isoT_rot_dg/post_equal_weights.dat")

pressures=np.logspace(p_minbar,p_maxbar,n_pressure)

abundances = {}

atmosphere_LRS = Radtrans(pressures=pressures,
                               line_species=[ 'H2O', 'CO-NatAbund', 'CO2'],
                               rayleigh_species=['H2', 'He'],
                               gas_continuum_contributors=['H2-H2', 'H2-He'],
                               wavelength_boundaries=[2.5, 5.5])



tdepth_fin = []
for i in range(len(post)):
    print(i)
#for i in range(4):
    para_dic = dict(MMR_H2O = post[i,2],T_eq = post[i,3],MMR_CO = post[i,4],MMR_CO2 = post[i,6],dg = post[i,7])#],Pcloud = post[i,7],haze = post[i,9],MMR_TiO=post[i,10],MMR_VO = post[i,11],MMR_Na=post[i,12],MMR_K=post[i,13])

    temperature = para_dic["T_eq"]*np.ones_like(pressures)
    Z= 10.0**para_dic["MMR_H2O"]+10.0**para_dic["MMR_CO"]+10.0**para_dic["MMR_CO2"]#+10.0**para_dic["MMR_TiO"]+10.0**para_dic["MMR_VO"]+10.0**para_dic["MMR_Na"]+10.0**para_dic["MMR_K"]


    MMR_H2 = (1.0-Z)*(1-HHe_ratio)

    MMR_He = HHe_ratio*(1.0-Z)

    mass_fractions_LRS = {'H2':MMR_H2* np.ones_like(temperature),
                               'He': MMR_He * np.ones_like(temperature),
                               'H2O': 10.0**para_dic["MMR_H2O"] * np.ones_like(temperature),
                               'CO-NatAbund': 10.0**para_dic["MMR_CO"] * np.ones_like(temperature),
                               'CO2': 10.0**para_dic["MMR_CO2"] * np.ones_like(temperature)}



    MMW = 1.0/(MMR_H2/2.0+MMR_He/4.0+10.0**para_dic["MMR_H2O"]/18.0+10.0**para_dic["MMR_CO2"]/44.0)*np.ones_like(temperature)


    wavelengths_LRS, transit_radii_LRS, _ = atmosphere_LRS.calculate_transit_radii(
    temperatures=temperature,
    mass_fractions=mass_fractions_LRS,
    mean_molar_masses=MMW,
    reference_gravity=gravity_SI*100*para_dic["dg"],
    planet_radius=radius,
    reference_pressure=P0_bar)


    wavelength_LR_microns = wavelengths_LRS*1e4
    radius_LR = transit_radii_LRS/100.0
    tdepth_LR = (radius_LR/Rs)**2*100.

    tdepth_fin.append(tdepth_LR)

tdepth_fin = np.array(tdepth_fin)
np.savetxt("/home/fdebras/tmp/tdepth_fin.txt",tdepth_fin)
np.savetxt("/home/fdebras/tmp/wl_fin.txt",wavelength_LR_microns)

