
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal



type_obs = "emission"

READ_DATA = True #do you want to read some t.fits files ?
INJ_PLANET = True
REDUCE_DATA = True #do you want to reduce one or several pkl file that has been read beforehand ?
CORREL_DATA = True #do you want to perform correlation from reduced pkl files ? 


dir_global = "/home/adminloc/Bureau/Atmospheres/Data/GL15A/"

### Directory to save figures if plot = True
dir_figures = dir_global+"Figures/"

num_obs = 1 #Number of observing nights that will be treated independently
#before being added up in the correlation



###########################################################################
###########################################################################
################### PARAMETERS TO READ DATA
###########################################################################
###########################################################################


### Directory where all the "t.fits" files are stores 
dir_data = [dir_global+"/fits/"]

### Name of the pickle file to store the info in 
dir_save_read = dir_global+"read/"
read_name_fin = ["GL15A_read.pkl"]


### List of SPIRou absolute orders -- Reddest: 31; Bluest: 79
orders   =  np.arange(31,80)[::-1].tolist() 
nord = len(orders)

### Ephemerides (to compute orbital phase)
T0       = 2459130.8962180                #Mid-transit (or cunjunction) time [BJD]
Porb     = 2.218577                      #Orbital period [d]
T_peri   = 2459130.8962180            ## Time of peri astron passage for an elliptical orbit
T_eclipse = T0+0.5*Porb #Our injected planet is on a circular orbit, in the case of non circular orbit 
                        #the eclipse time is not at half orbital period compared to periastron, unless periastron=transit

transiting  = False  ## Only used in emission, to calculate the window function during eclipse if need be


### Transit parameters -- Compute the transit window
### Using batman python package https://lweb.cfa.harvard.edu/~lkreidberg/batman/
### Get the limb-darkening coefficients in H band from Claret+2011: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/529/A75
Rp       = 0.5485*69911000. #Planet radius  [m]
Rs       = 0.375*696340000.  #Stellar radius [m] 
Ms       = 0.4*1.989*1e30     #Stellar mass [kg] 
ip       = 90.0    #Transit incl.  [deg]
ap       = 14.0534030   #Semi-maj axis  [R_star]
ep       = 0.    #Eccentricity of Pl. orbit
wp       = 0.0     #Arg of periaps [deg]
ld_mod   = "quadratic"     #Limb-darkening model ["nonlinear", "quadratic", "linear"]
ld_coef  = [0.0156,0.313]  #Limb-darkening coefficients 

T_star = 3600.  #stellar temperature [K]. Only necessary for injecting a planet in emission.
### Stellar radial velocity info
Ks        = 0.164    #RV semi-amplitude of the star orbital motion due to planet [km/s]
V0        = 11.73    #Stellar systemic velocity [km/s]



### Plots
plot_read     = True     # If True, plot transit info
figure_name_transit = ["transit_GL15A"] #name of the figure file


###########################################################################
###########################################################################
################### PARAMETERS TO INJECT PLANET 
###########################################################################
###########################################################################


planet_wavelength_nm_file = "/home/adminloc//Bureau/Atmospheres/Pipeline_v2/ATMOSPHERIX_DATA_RED/Models/Results/lambdastest-GL15A_onlyH2O.txt"

# Radius (in meters) for transit. CAREFUL : it is not the transit depth, it is indeed radius
planet_radius_m_file = "/home/adminloc/Bureau/Atmospheres/Pipeline_v2/ATMOSPHERIX_DATA_RED/Model/Results/RpGL15A_HD189_onlyH2O-VMR3-T900.txt"

#planetary flux in J/m-2/s-1
planet_flux_SI_file = "/home/adminloc//Bureau/Atmospheres/Pipeline_v2/ATMOSPHERIX_DATA_RED/Models/Results/fluxtest-GL15A_onlyH2O.txt"
K_inj = 120.
V_inj = 10.
amp_inj = 1





###########################################################################
###########################################################################
################### PARAMETERS TO REDUCE DATA
###########################################################################
###########################################################################

dir_reduce_in = dir_global+"read/"
dir_reduce_out = dir_global+"reduced/"
reduce_name_in = ["GL15A_read.pkl",]
#output fil
reduce_name_out  = ["GL15A_reduced.pkl",]
#information file
reduce_info_file = dir_figures+"info.dat"


### Correction of stellar contamination
### Only used if synthetic spectrum available
corr_star  = False
WC_name    = ""            ### Input wavelength for synthetic stellar spectra
IC_name    = ""            ### Input flux for synthetic stellar spectra

### Additional Boucher correction. If dep_min >=1, not included. 
dep_min  = 0.7  # remove all data when telluric relative absorption > 1 - dep_min
thres_up = 0.1      # Remove the line until reaching 1-thres_up
Npt_lim  = 800      # If the order contains less than Npt_lim points, it is discarded from the analysis

### Interpolation parameters
pixel    = np.linspace(-0.5,0.5,15)   ### Sampling a SPIRou pixel in velocity space -- Width ~ 2.28 km/s
sig_g    = 2.28                       ### STD of one SPIRou px in km/s
N_bor    = 15                           ### Nb of pts removed at each extremity (twice)

### Normalisation parameters. We normalize once before calculating the master spectrum, and once afterwards
### theoretically to remove nodal noise
first_norm_type = "percentile"
second_norm_type = "old"

N_med    = 150                          ### Nb of points used in the median filter for the inteprolation
sig_out  = 5.0                          ### Threshold for outliers identification during normalisation process 
N_adj = 2 ### Number of adjacent pixel removed with outliers
deg_px   = 2                            ### Degree of the polynomial fit to the distribution of pixel STDs

### Parameters for detrending with airmass
det_airmass = False
deg_airmass = 2

### Parameters PCA. Auto-tune automatically decides the number of component 
#to remove by comparing with white noise map.
mode_pca    = "pca"                     ### "pca" or "autoencoder"
wpca = False   #Use weighted pca
auto_tune   = True                  ### Automatic tuning of number of components
factor_pca = 1. #actor in the auto tune: every PC above factor*white_noise_mean_eigenvalue is suppressed
min_pca  = 1 #minimim number of removed components
mode_norm_pca = "global" #how to remove mean and std in the data before PCA. Four possibilities:
                         # "none" : data untouched.
                         # "global" : suppression of mean and division by the std of the whole data set 
                         # 'per_pix': same as global but column by colum (per pixel)
                         # 'per_obs': same as global but line by line (per observation)
            

 ### Nb of removed components if auto tune is false
npca        = np.array(1*np.ones(49),dtype=int)     


### Plot info
plot_red    = True
numb        = 46
    

#If you want to remove some orders, put them here
orders_rem     = []


#We can manually decide the where is the transit in the phase direction,
#and exclude it for the calculation of the mean stellar spectrum.
#If set_window = False, the transit window defines n_ini and n_end
set_window = True
n_ini_fix,n_end_fix = -1,-1    ### Get transits start and end indices. If you want everything, put -1 -1 e.g. in emission

### Size of the estimation of the std of the order for final metrics
N_px          = 200




###########################################################################
###########################################################################
################### PARAMETERS FOR CORRELATION
###########################################################################
############################################################################

parallel = False
#This is just for the integration over a pixel
pixel_correl = np.linspace(-0.5,0.5,15)
weights= np.ones(15)

#Kp intervals
Kpmin = 0.0
Kpmax =200.0
Nkp = 81
Kp_array = np.linspace(Kpmin,Kpmax,Nkp)

#Vsys intervals
Vmin = -20.
Vmax= 20
Nv = 81
Vsys_array = np.linspace(Vmin,Vmax,Nv)

#Number of pkl observations files and their names

dir_correl_in = dir_global+"reduced/"

# correl_name_in = ["HD189_Sep18.pkl",]
correl_name_in = ["GL15A_reduced.pkl"]



#Do we save the correlation file ? If yes, put as much files as there are observations
save_ccf = True
dir_correl_out = dir_global+"correlated/"

# correl_name_out = ["HD189_Sep18.pkl",]
correl_name_out = ["GL15A_correlated.pkl"]


dir_correl_mod = "/home/adminloc/Bureau/Atmospheres/Pipeline_v2/ATMOSPHERIX_DATA_RED/Templates/GL15A_example_onlyH2O/"

#DO we select orders or take them all ? If True, provide your order selection
# for each observation. If an order does not exist in the pkl file, it will 
# obivously not be used but will not trigger an error.
select_ord = False
list_ord_correl = np.arange(32,34) #not used if select_ord = False



#If false, the calculation is performed over the whole dataset. If 
#True, we only select observation that have a transit window > min_window
select_phase = True
min_window = 0.2

#Interpolation factor for the speed array. If you d'ont know what that means, choose something between 1 and 10
int_speed = 5

#Number of pixels to discard at the borders. 
nbor_correl = 10

#Do we include the projector from Gibson 2022 ?
use_proj = False
#If we just removed the mean and std of the whole map, we can use a fast verion of the projector
#Else, it will be even longer
proj_fast = False
mode_norm_pca_correl = "per_obs" #if proj_fast is not used, we can choose
                          #how to remove mean and std in the data before PCA. Four possibilities:
                          # "none" : data untouched.
                          # "global" : suppression of mean and division by the std of the whole data set 
                          # 'per_pix': same as global but column by colum (per pixel)
                          # 'per_obs': same as global but line by line (per observation)

#Do we select only certain orders for the plot ? 
#if yes, lili is the list oforders to select
select_plot = False
list_ord_plot_correl = np.array([48,47,46,34,33,32])

#In order to calculate the std of the map,we need to exclude 
#a zone of the Kp-Vsys map around the planet. These are the limits 
#of this rectangular zone.
Kp_min_std = 50
Kp_max_std = 200
Vsys_min_std = 0
Vsys_max_std = 20

#number of levels in the contour plot
nlevels = 15

#Do we plot the correlation map at each obs ?
plot_ccf_indiv = True
#Do we plot the global correaltion map ? 
plot_ccf_tot = True


#Do we add white lines at the planet position ? 
white_lines = True
Kp_planet = 120.
Vsys_planet = 10.





###########################################################################
###########################################################################
################### PARAMETERS FOR PLOTS
###########################################################################
############################################################################
SMALL_SIZE = 28
MEDIUM_SIZE = 32
BIGGER_SIZE = 34
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title




