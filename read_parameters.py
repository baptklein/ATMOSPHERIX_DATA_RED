
import numpy as np

pipeline_rep = "/home/florian/Bureau/Atmosphere_SPIRou/Pipeline_git/"

### Directory where all the "t.fits" files are stores 
dir_data = pipeline_rep+"Data/T_files"

### Directory to save figures if plot = True
dir_figures = pipeline_rep+"Figures/"

### Name of the pickle file to store the info in 
name_fin = pipeline_rep+"test.pkl"

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


### Stellar radial velocity info
Ks        = 0.171    #RV semi-amplitude of the star orbital motion due to planet [km/s]
V0        = 11.73    #Stellar systemic velocity [km/s]

### Plots
plot      = True     # If True, plot transit info







