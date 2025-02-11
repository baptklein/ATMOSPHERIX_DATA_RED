#!/bin/bash
#SBATCH -J C-J_HIP67522
#SBATCH --partition=spirou
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --ntasks-per-core=1
#SBATCH --time=24:00:00
#SBATCH -o test.out
#SBATCH -e test.err
DIR='/home/fdebras/Multinest_from_calmip/Multinest_HIP67522/'
cd /home/fdebras/ 
source ~/.bashrc
#source ~/multinest-env.bash
#export LD_LIBRARY_PATH=~/multinest/lib:/home/soft/intel/mkl/lib/intel64_lin/:$LD_LIBRARY_PATH
#export PATH=$PATH:$HOME/.local/bin/
#export LD_PRELOAD=/home/soft/intel/mkl/lib/intel64_lin/libmkl_core.so:/home/soft/intel/mkl/lib/intel64_lin/libmkl_sequential.so
which mpirun
cd $DIR
export OMP_NUM_THREADS=1
mpirun -np  20 python3 multinest_atmo.py --like Gibson_transit --data data.py --winds 
