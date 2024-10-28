
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal



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



READ_DATA = True
REDUCE_DATA = False
CORREL_DATA = False


dir_global = "/home/florian/Bureau/Atmosphere_SPIRou/Pipeline_git/Data/"

### Directory to save figures if plot = True
dir_figures = dir_global+"Figures/"

num_obs = 1 #Number of observing nights that will be treated independently


################### PARAMETERS TO READ DATA
### Directory where all the "t.fits" files are stores 

dir_data = [dir_global+"T_files/"]

### Name of the pickle file to store the info in 
dir_save_read = dir_global+"read/"
read_name_fin = ["test.pkl"]

### List of SPIRou absolute orders -- Reddest: 31; Bluest: 79
orders   =  np.arange(31,80)[::-1].tolist() 
nord = len(orders)

### Ephemerides (to compute orbital phase)
T0       =  2459130.8962180                #Mid-transit time [BJD]
Porb     = 2.218577                      #Orbital period [d]

### Transit parameters -- Compute the transit window
### Using batman python package https://lweb.cfa.harvard.edu/~lkreidberg/batman/
### Get the limb-darkening coefficients in H band from Claret+2011: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/A+A/529/A75
Rp       = 39849.27 #Planet radius  [km]
Rs       = 270179.920  #Stellar radius [km] 
ip       = 90.0    #Transit incl.  [deg]
ap       = 13.58   #Semi-maj axis  [R_star]
ep       = 0.0     #Eccentricity of Pl. orbit
wp       = 0.0     #Arg of periaps [deg]
ld_mod   = "quadratic"     #Limb-darkening model ["nonlinear", "quadratic", "linear"]
ld_coef  = [0.0156,0.313]  #Limb-darkening coefficients 


inj_planet = True
amp_inj = 1

### Stellar radial velocity info
Ks        = 0.171    #RV semi-amplitude of the star orbital motion due to planet [km/s]
V0        = 11.73    #Stellar systemic velocity [km/s]

### Plots
plot_read     = True     # If True, plot transit info



######################" PARAMETERS TO REDUCE DATA

dir_reduce = [dir_global+"reduced/"]
reduce_name_in = [dir_reduce+'test_withp.pkl']
#output fil
reduce_name_out  = [dir_reduce+'test_new.pkl' ]
#information file
nam_info = dir_reduce+"info.dat"


### Correction of stellar contamination
### Only used if synthetic spectrum available
corr_star  = False
WC_name    = ""            ### Input wavelength for synthetic stellar spectra
IC_name    = ""            ### Input flux for synthetic stellar spectra

### Additional Boucher correction. If dep_min >=1, not included. 
dep_min  = 1.0  # remove all data when telluric relative absorption > 1 - dep_min
thres_up = 0.05      # Remove the line until reaching 1-thres_up
Npt_lim  = 800      # If the order contains less than Npt_lim points, it is discarded from the analysis

### Interpolation parameters
pixel    = np.linspace(-1.14,1.14,11)   ### Sampling a SPIRou pixel in velocity space -- Width ~ 2.28 km/s
sig_g    = 2.28                       ### STD of one SPIRou px in km/s
N_bor    = 15                           ### Nb of pts removed at each extremity (twice)

### Normalisation parameters
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
factor_pca = 1. #factor in the auto tune: every PC above factor*white_noise_mean_eigenvalue is suppressed
mode_norm_pca = "none" #how to remove mean and std in the data before PCA. Four possibilities:
                         # "none" : data untouched.
                         # "global" : suppression of mean and division by the std of the whole data set 
                         # 'per_pix': same as global but column by colum (per pixel)
                         # 'per_obs': same as global but line by line (per observation)

 ### Nb of removed components if auto tune is false
npca        = np.array(1*np.ones(49),dtype=int)     


### Plot info
plot_red    = True
numb        = 46
nam_fig     = dir_figures+"reduc_" + str(numb) + ".png"
    

#If you want to remove some orders, put them here
orders_rem     = []


#We can manually decide the where is the transit in the phase direction,
#and exclude it for the calculation of the mean stellar spectrum.
#If set_window = False, the transit window defines n_ini and n_end
set_window = False
n_ini_fix,n_end_fix = 10,20    ### Get transits start and end indices

### Size of the estimation of the std of the order for final metrics
N_px          = 200




########################### PARAMETERS FOR CORRELATION
#Do you want to run it in parallel ?
parallel = False
#This is just for the integration over a pixel
c0 = 299792.458
pixel_window = np.linspace(-1.17,1.17,15)
weights = scipy.signal.gaussian(15,std=1.17)
weights= np.ones(15)

#Kp intervals
Kpmin = 0.0
Kpmax =300.0
Nkp = 61
Kp = np.linspace(Kpmin,Kpmax,Nkp)

#Vsys intervals
Vmin = -150
Vmax= 150
Nv = 301
Vsys = np.linspace(Vmin,Vmax,Nv)

#Number of pkl observations files and their names

dir_correl = [pipeline_rep+]


file_list = [pipeline_rep+"test_new.pkl",]
file_list = ["/home/florian/Bureau/Atmosphere_SPIRou/Data/HD189/pkl/reduced/DRS_06_0721/Jun19/"+"HD189_Jun19_new-pipeline_boucher05_autoPCA-1.pkl",\
             "/home/florian/Bureau/Atmosphere_SPIRou/Data/HD189/pkl/reduced/DRS_06_0721/Sep18/"+"HD189_Sep18_new-pipeline_boucher05_autoPCA-1.pkl"]


dire_mod = "/home/florian/Bureau/Atmosphere_SPIRou/Pipeline_git/Data_Simulator/Model/Results/to-correl/reducedGL15A_HD189_onlyH2O-VMR3-T900/"


#DO we select orders or take them all ? If True, provide your order selection
# for each observation. If an order does not exist in the pkl file, it will 
# obivously not be used but will not trigger an error.
select_ord = True
list_ord1 = np.arange(32,80)

#If false, the calculation is performed over the whole dataset. If 
#True, we only select observation that have a transit window > min_window
select_phase = True
min_window = 0.2

#Interpolation factor for the speed array. If you d'ont know what that means, choose something between 1 and 10
int_speed = 8

#Number of pixels to discard at the borders. 
nbor = 10

#Do we include the projector from Gibson 2022 ?
use_proj = True
#If we just removed the mean and std of the whole map, we can use a fast verion of the projector
#Else, it will be even longer
proj_fast = True
mode_norm_pca = "per_obs" #if proj_fast is not used, we can choose
                          #how to remove mean and std in the data before PCA. Four possibilities:
                          # "none" : data untouched.
                          # "global" : suppression of mean and division by the std of the whole data set 
                          # 'per_pix': same as global but column by colum (per pixel)
                          # 'per_obs': same as global but line by line (per observation)

#Do we select only certain orders for the plot ? 
#if yes, lili is the list oforders to select
select_plot = False
lili = np.array([48,47,46,34,33,32])

#In order to calculate the std of the map,we need to exclude 
#a zone of the Kp-Vsys map around the planet. These are the limits 
#of this rectangular zone.
Kp_min_std = 80
Kp_max_std = 160
Vsys_min_std = 20
Vsys_max_std = 40

#number of levels in the contour plot
nlevels = 15

#Do we plot the correlation map at each obs ?
plot_ccf_indiv = True
#Do we plot the global correaltion map ? 
plot_ccf_tot = True

#Do we save the correlation file ? If yes, put as much files as there are observations
save_ccf = False
filesave_list = [pipeline_rep+"correlated.pkl"]

#Do we add white lines at the planet position ? 
white_lines = True
Kp_planet = 151
Vsys_planet = -4.5


