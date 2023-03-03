
import numpy as np
from petitRADTRANS import Radtrans
import pylab as plt
from petitRADTRANS import nat_cst as nc


dire_res = "../Results/"
suffix_res = "GL15A_HD189_onlyH2O-VMR3-T900"

pressures = np.logspace(-8, 2, 101)

Z1_name = 'H2O_main_iso'
Z1_mass = 18
Z1_VMR = 1e-3*np.ones_like(pressures)

Z_VMR = Z1_VMR
Z_mass = Z1_VMR*Z1_mass

nhe = (1.-Z_VMR)/(1.+1./2.*(4./0.275-4.))
nh2 = (4./0.275-4.)*nhe/2.

MMW = (Z_mass+2.*nh2+4*nhe)
X = 2.*nh2/MMW
Y = 4.*nhe/MMW
Z1 = Z1_VMR*Z1_mass/MMW

atmosphere = Radtrans(line_species = [Z1_name],\
#,'Fe','FeH_main_iso','Na','K','CO_main_iso'], \
      rayleigh_species = ['H2', 'He'], \
      continuum_opacities = ['H2-H2','H2-He'], \
      wlen_bords_micron = [0.9,2.6], \
      lbl_opacity_sampling = 4, \
      mode = 'lbl')


R_pl = 0.55*nc.r_jup_mean
gravity = 4918.0 #in cm.s-2
P0 = 0.1

atmosphere.setup_opa_structure(pressures)
kappa_IR = 0.01
gamma = 0.012
T_int = 200.
T_equ = 900.0
#temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
temperature = T_equ*np.ones_like(pressures)

abundances = {}
abundances['H2'] = X* np.ones_like(temperature)
abundances['He'] = Y* np.ones_like(temperature)
abundances[Z1_name] =Z1* np.ones_like(temperature)

atmosphere.calc_transm(temperature, abundances, gravity, MMW, R_pl=R_pl, P0_bar=P0)


np.savetxt(dire_res+'lambdas'+suffix_res+'.txt',nc.c/atmosphere.freq/1.0e-7)
np.savetxt(dire_res+'Rp'+suffix_res+'.txt',atmosphere.transm_rad)



