# ATMOSPHERIX_DATA_RED
We provide a sample tool to read SPIRou "t.fits" files, process the data (remove telluric/stella contributions)
and make the correlation analysis for a given planet atmosphere template.

Requirements: Python 3 modules
- scipy
- pickle
- sklearn: https://scikit-learn.org/
- batman: https://lweb.cfa.harvard.edu/~lkreidberg/batman/ . Be very careful !!! Don't do pip install batman but pip install batman-package
- astropy

If you want to run the nested sampling algorithm, you will also need pymultinest (and necessarily multinest). We refer you to pymultinest documentation: https://johannesbuchner.github.io/PyMultiNest/) 

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import time



### To run the code:
1. Going from a list of "t.fits" files to the format compatible with our data reduction code:
    - Read fits files
    - Correct for Blaze and remove NaNs from the observations
    - Compute orbital phase and transit window
    - Store pre-processed data in ".pkl" files
   
  Test case: You can download observations of Gl 15 A and the associated models via the following repository:
  https://drive.google.com/drive/folders/1CHupP3I4r6bzJgUoblpmi8SKpEQvRnEs?usp=sharing
   
  To apply the code, change the paramters in the file "read_data.py" and type the following command line:
    
      $ python read_data.py
      
  If you only do that however you will not have a planet in your data as GL 15A has no known transiting planets.  You can instead add a synthetic HD 189733 with a temperature of 900K and a water VMR of 0.001 using the Jupyter notebook in Data_Simulator, that loads the Model in Data_Simulator/Model/Results/. Both methods (read_data or main.ipynb) will give you a pkl file to use in the following.
  
  Instead of reading point 2, 3 and 4 you can also keep using the notebook until the correlation part (that does not work in the notebook). It uses an old version of the data pipeline but shows you graphically all the steps.
      

2. Data reduction process: change parameters in "reduce_data.py" and run

      $ python reduce_data.py
   
   
3. Compute the correlation between the data and model for a grid of planet velocimetric semi-amplitude and systemic velocity:

  Change parameters in "get_correl.py" and run
  
    $ python get_correl.py
    
  WARNING: get_correl is much worse than Correlate.py but more user friendly. If you have time or are one of Florian's interns, use Correlate.
  
  This will use the templates in Data_Simulator/Model/results/to-correl, created by the file create_templates in Template_generator. If you wish to use different planets you will obviously need to create new templates. 
 
 
4. The instructions to run the nested sampling are in the multinest_atmo folder. Have fun !
