# ATMOSPHERIX_DATA_RED
We provide a sample tool to read SPIRou "t.fits" files, process the data (remove telluric/stella contributions)
and make the correlation analysis for a given planet atmosphere template. The current version only takes into account
transmission spectroscopy, but emission will soon be publicly available.

Requirements: Python 3 modules
- scipy
- pickle
- sklearn: https://scikit-learn.org/
- batman: https://lweb.cfa.harvard.edu/~lkreidberg/batman/ . Be very careful !!! Don't do pip install batman but pip install batman-package
- astropy
- wpca (not necessary but recommended)

If you want to run the nested sampling algorithm, you will also need petitRADTRANS and pymultinest (and necessarily multinest). We refer you to pymultinest documentation: https://johannesbuchner.github.io/PyMultiNest/) 

We provide a sample of t.fits files for an observing sequence of GJ15A, as used in the ATMOSPHERIX  I and II papers:
  https://drive.google.com/drive/folders/1CHupP3I4r6bzJgUoblpmi8SKpEQvRnEs?usp=sharing


### To run the code:
The code is run by the main.py file, which can read, reduce and/or correl the data depending on your choice.
You can also inject a synthetic planet. 
The parameters are in the parameters.py file. 

The easiest way to run the example in the code is to created a directory, that will be your global directory. 
Inside that directory, create a "fits", a "read", a "reduced", a "correlated" and a "Figures" directory. 
The "fits" directory will contain all the GL15A observation, and the read, reduced and correlated directory the pkl files after the steps of the pipeline.
Finally, the "Figures" directory contains the figures if you want to plot stuff during analysis. 

To read the data you need to indicate the location of the t.fits files in the 
"dir_data" parameter array (see previous paragraph). You can have several observation sequences, and in that case pay attention
that dir_data must be of length (num_obs). Once read, the data are stored in pkl files to which you need to provide
names in the parameter file. In detail, the "read" function:
    - Reads fits files
    - Corrects for Blaze and remove NaNs from the observations
    - Computes orbital phase and transit window
    - Stores pre-processed data in ".pkl" files

Regarding data reduction, all the parameters and what they do is detailed in both the parameters.py file 
and most importantly in the ATMOSPHERIX 1 paper. At the end of the reduction process, datas are stored 
in another pkl file that can either be used for correlation or multinest.
   
   
Finally, the correlation function computes the correlation between the data and model for a 
grid of planet velocimetric semi-amplitude and systemic velocity. This function uses 
the templates in Data_Simulator/Model/results/to-correl, created by the file create_templates in Template_generator. 
If you wish to use different planets you will obviously need to create new templates. 
 

The instructions to run the nested sampling are in the multinest_atmo folder, but are not up to date yet. 
Don't hesitate to contact us. Have fun !!
