#!/bin/sh
# -N atmo
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=32 
#SBATCH -n 32
#SBATCH --job-name=broad_boucher 
#SBATCH -p n27
#SBATCH -o test.out
#SBATCH -e test.err
DIR='/home/fdebras/Multinest_from_calmip/Multinest_HD189/'
source ~/.bashrc
source ~/multinest-env.bash
export LD_LIBRARY_PATH=~/multinest/lib:/home/soft/intel/mkl/lib/intel64_lin/:$LD_LIBRARY_PATH
export PATH=$PATH:$HOME/.local/bin/
export LD_PRELOAD=/home/soft/intel/mkl/lib/intel64_lin/libmkl_core.so:/home/soft/intel/mkl/lib/intel64_lin/libmkl_sequential.so
which mpirun
cd $DIR
export OMP_NUM_THREADS=1
mpirun -np  32 python3 multinest_atmo.py --like Gibson_transit --data data.py --winds
