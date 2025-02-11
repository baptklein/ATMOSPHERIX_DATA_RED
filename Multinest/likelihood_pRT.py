from petitRADTRANS.retrieval.data import Data
import numpy as np
import model_pRT as mdl #you need this python file to calclate de model
import os

#file with the observed data
data_file = "new_full_WASP-127b_DATA.ecsv" 

#Function that gives the likelihood calculated with petitRDATRANS
def likelihood(data_file):
        """
        Calculate the log-likelihood between the model and the data. Using the log_likelihood_gibson function of the retrieval (petitRADTRANS)

        The spectrum model must be on the same wavelength grid than the data.
        Args:
            model: numpy.ndarray
                The model flux in the same units as the data.
            data: numpy.ndarray
                The data.
            uncertainties: numpy.ndarray
                The uncertainties on the data.
            alpha: float, optional
                Model scaling coefficient.
            beta: float, optional
                Noise scaling coefficient. If None,
        Returns:
            logL :
                The log likelihood of the model given the data.
        """        
        uncertainties = []
        data = []
        data_wavelenght = []
        data_table = np.loadtxt(data_file, skiprows=1)
        for i in range(len(data_table)): #reading the data file
            data_wavelenght.append(data_table[i][0])
            data.append(data_table[i][1])
            uncertainties.append(data_table[i][2])
       
        
        model_wavelenght, model, T, Rpl, g = mdl.best_model() #taking the model values
        
           
        data = np.array(data)
        uncertainties = np.array(uncertainties)

        #taking the same wavelenght points for both data and model
        new_model = []
        eps = 0.0004
        for i in range (len(model_wavelenght)):
            for j in range(len(data_wavelenght)):
                if ((model_wavelenght[i] <= data_wavelenght[j] + eps) and (model_wavelenght[i] >= data_wavelenght[j] - eps)).all():
                    new_model.append(model[i])
        
        #convert to array
        new_model = np.array(new_model)


        L = Data.log_likelihood_gibson(new_model,data,uncertainties)
        return (L)

print("La vraisemblance L vaut : " + str(likelihood(data_file)))
print("Vous trouverez le spectre de transmission du modèle sous : retrieval_" + mdl.name_of_my_retrieval)
