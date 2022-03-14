# ATMOSPHERIX_DATA_RED
We provide a sample tool to read SPIRou "t.fits" files, process the data (remove telluric/stella contributions)
and make the correlation analysis for a given planet atmosphere template.

Requirements: Python 3 modules
- scipy
- pickle
- sklearn: https://scikit-learn.org/
- batman: https://lweb.cfa.harvard.edu/~lkreidberg/batman/
- astropy


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
  To apply the code, change the paramters in the file "read_data.py" and type the following command line:
    
      $ python read_data.py
      

2. Data reduction process: change parameters in "reduce_data.py" and run

      $ python reduce_data.py
      
   Test case: You can download observations of Gl 15 A (in the .pkl format) and the associated models via the following repository:
   https://drive.google.com/drive/folders/1eWhGpNrjLUSoyWKbWYVv15o0UZ0Ucn6D?usp=sharing
   
   
3. Compute the correlation between the data and model for a grid of planet velocimetric semi-amplitude and systemic velocity:

  Change parameters in "get_correl.py" and run
  
    $ python get_correl.py
