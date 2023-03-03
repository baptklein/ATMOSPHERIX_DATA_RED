To run the pymultinest based nested sampling algorithm, you need to follow these few steps:

1) Start from a reduced pkl file (from the data analysis)
2) Run reduced-to-mcmc.py to create a nested sampling-ready pkl file
3) Update the data.py file with your planet, orders and wavelengths (if you are not using SPIRou)
4) Run the sampler : mpirun -np N python multinest_atmo.py --data data.py --like Brogi/Gibson
5) Do not use the --winds option thus far, it is full of bugs

To update the prior, or the other parameters, you have to enter into the code for now (multinest_atmo.py and petit_model.py) . It is being updated. 

Unfortunately, github does not allow us to put a >25Mb file on the repository so we cannot provide an example pkl file to run the sampler, but it is very easy to create one from our fabulous data reduction code. 



**SIDENOTE**

If you want to download the test pkl files, you will need the git large file storage software. See the documentation here : 
https://docs.github.com/en/repositories/working-with-files/managing-large-files
