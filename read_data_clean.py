"""
Created on 03/2022 12:54:11 2021

This script performs data processing and analysis for a transit observation.
Authors: Baptiste & Florian
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import read_functions as read_func
from read_parameters import *

"""
This code reads and analyzes transit observation data.
"""
### READ FITS FILES
print("\nRead data from", dir_data)
list_ord = [read_func.Order(order) for order in orders]  # Initialize list of Order objects
list_ord, airmass, T_obs, berv, snr_mat = read_func.read_data_spirou(dir_data, list_ord, nord)
nobs = len(T_obs)
print("DONE")

### Pre-process data: correct for blaze and remove NaNs
print("\nRemove NaNs")
list_ord = [o for o in list_ord if not o.remove_nan()]
nord = len(list_ord)
print("DONE")

print("\nCompute transit")
### Compute phase
phase = (T_obs - T0) / Porb
phase -= int(phase[-1])

### Compute transit window
flux = read_func.compute_transit(Rp, Rs, ip, T0, ap, Porb, ep, wp, ld_mod, ld_coef, T_obs)
window = (1 - flux) / np.max(1 - flux)
print("DONE")

### Compute Planet-induced RV
Vp = read_func.get_rvs(T_obs, Ks, Porb, T0)
Vc = V0 + Vp - berv  # Geocentric-to-barycentric correction

### Plot transit information
if plot:
    print("\nPlot transit")
    TT = 24. * (T_obs - T0)
    ypad = 15  # pad of the y label
    plt.figure(figsize=(15, 12))

    plt.subplot(411)
    plt.plot(TT, flux, "-+r", label="HD 189733 b analog")
    plt.legend(loc=3, fontsize=16)
    plt.ylabel("Transit curve\n", labelpad=ypad)

    plt.subplot(412)
    plt.plot(TT, airmass, "-k")
    plt.ylabel("Airmass\n", labelpad=ypad)

    plt.subplot(413)
    plt.plot(TT, Vc, "-k")
    plt.ylabel("RV correction\n[km/s]", labelpad=ypad)

    plt.subplot(414)
    plt.plot(TT, np.max(snr_mat, axis=1), "+k")
    plt.axhline(np.mean(np.max(snr_mat, axis=1)), ls="--", color="gray")
    plt.xlabel("Time wrt transit [h]")
    plt.ylabel("Peak S/N\n", labelpad=ypad)

    plt.subplots_adjust(hspace=0.02)
    plt.savefig(dir_figures+"transit_info.pdf", bbox_inches="tight")
    plt.close()
    print("DONE")

### Save as pickle
print("\nData saved in", name_fin)
data_to_save = (orders, [o.W_raw for o in list_ord], [o.I_raw for o in list_ord],
                [o.B_raw for o in list_ord], [o.I_atm for o in list_ord],
                T_obs, phase, window, berv, V0 + Vp, airmass, snr_mat)
with open(name_fin, 'wb') as specfile:
    pickle.dump(data_to_save, specfile)
print("DONE")
