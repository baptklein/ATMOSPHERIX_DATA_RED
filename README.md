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

If you want to run the nested sampling algorithm, you will also need petitRADTANS and pymultinest (and necessarily multinest). We refer you to pymultinest documentation: https://johannesbuchner.github.io/PyMultiNest/) 

We provide a sample of t.fits files for an observing sequence of GJ15A, as used in the ATMOSPHERIX  I and II papers:
  https://drive.google.com/drive/folders/1CHupP3I4r6bzJgUoblpmi8SKpEQvRnEs?usp=sharing


### To run the code:
The code is run by the main.py file, which can read, reduce and/or correl the data depending on your choice.
You can also adda synthetic planet. 
The parameters are in the parameters.py file. 

To read the data you need to indicate the location of the t.fits file in the 
"dir_data" parameter array. You can have several observation sequences, and in that case pay attention
that dir_data must be of length (num_obs). Once read, the data are stored in pkl files to which you need to provide
names in the parameter file. In detail, the "read" function:
    - Reads fits files
    - Corrects for Blaze and remove NaNs from the observations
    - Computes orbital phase and transit window
    - Stores pre-processed data in ".pkl" files
   


  If you only do that however you will not have a planet in your data as GL 15A has no known transiting planets.  You can instead add a synthetic HD 189733 with a temperature of 900K and a water VMR of 0.001 using the Jupyter notebook in Data_Simulator, that loads the Model in Data_Simulator/Model/Results/. Both methods (read_data or main.ipynb) will give you a pkl file to use in the following.
  
  Instead of reading point 2, 3 and 4 you can also keep using the notebook until the correlation part (that does not work in the notebook). It uses an old version of the data pipeline but shows you graphically all the steps.
      

2. Data reduction process: change parameters in "reduce_parameters.py" and run

      $ python reduce_data.py
   
   
3. Compute the correlation between the data and model for a grid of planet velocimetric semi-amplitude and systemic velocity:

  Change parameters in "correlate_parameters.py" and run
  
    $ python correlate.py
      
  This will use the templates in Data_Simulator/Model/results/to-correl, created by the file create_templates in Template_generator. If you wish to use different planets you will obviously need to create new templates. 
 

4. The instructions to run the nested sampling are in the multinest_atmo folder, but are not up to date yet. Don't hesitate to contact us. Have fun !!
