import matplotlib.pyplot as plt
import numpy as np

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.physics import temperature_profile_function_guillot_global

dire_res = "../Results/"
suffix_res = "test-GL15A_onlyH2O"


planet_radius = 1.0 * cst.r_jup_mean
reference_gravity = 10 ** 3.5
reference_pressure = 0.01


infrared_mean_opacity = 0.01
gamma = 0.4
intrinsic_temperature = 200
equilibrium_temperature = 1500

pressures=np.logspace(-10,2,130),

Z1_name = '1H2-16O'
Z1_mass = 18
Z1_VMR = 1.e-3*np.ones_like(pressures)

Z2_name = '12C-16O'
Z2_mass = 28.0
Z2_VMR = 0.e-0*np.ones_like(pressures)

Z3_name = 'CH4_main_iso'
Z3_mass = 16.0
Z3_VMR = 0*10**(-3.0)*np.ones_like(pressures)

Z4_name = 'NH3_main_iso'
Z4_mass = 17.0
Z4_VMR = 0.0*10**(-4.0)*np.ones_like(pressures)



Z_VMR = Z1_VMR+Z2_VMR+Z3_VMR+Z4_VMR
Z_mass = Z1_VMR*Z1_mass+Z2_VMR*Z2_mass+Z3_VMR*Z3_mass+Z4_VMR*Z4_mass

nhe = (1.-Z_VMR)/(1.+1./2.*(4./0.275-4.))
nh2 = (4./0.275-4.)*nhe/2.

MMW = (Z_mass+2.*nh2+4*nhe)
X = 2.*nh2/MMW
Y = 4.*nhe/MMW
Z1 = Z1_VMR*Z1_mass/MMW
Z2 = Z2_VMR*Z2_mass/MMW
Z3 = Z3_VMR*Z3_mass/MMW
Z4 = Z4_VMR*Z4_mass/MMW


mass_fractions = {
    'H2':X,
    'He': Y,
    Z1_name: Z1,
    Z2_name: Z2
}



mean_molar_masses = MMW


 

atmosphere = Radtrans(
    pressures=np.logspace(-10,2,130),
    line_species=[
        Z1_name,
        Z2_name
    ],
    rayleigh_species=['H2', 'He'],
    gas_continuum_contributors=['H2-H2', 'H2-He'],
    wavelength_boundaries=[0.8, 2.6],
    line_opacity_mode='lbl',
    line_by_line_opacity_sampling=4, # M
)

print(atmosphere.pressures[-1])
pressures = atmosphere.pressures*1e-6 # cgs to bar

temperatures = temperature_profile_function_guillot_global(
    pressures=pressures,
    infrared_mean_opacity=infrared_mean_opacity,
    gamma=gamma,
    gravities=reference_gravity,
    intrinsic_temperature=intrinsic_temperature,
    equilibrium_temperature=equilibrium_temperature
)


wavelength,flux, _ = atmosphere.calculate_flux(
    temperatures=temperatures,
    mass_fractions=mass_fractions,
    mean_molar_masses = mean_molar_masses,
    reference_gravity = reference_gravity,
)



np.savetxt(dire_res+'lambdas'+suffix_res+'.txt',wavelength*1.0e7) #from cm to nm
np.savetxt(dire_res+'flux'+suffix_res+'.txt',flux*0.1) #from erg.cm-2.s-1.cm-1 to J.m-2.s-1.m-1 (10-7*10 6)




