#!/bin/sh
# -N atmo
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH -n 1
#SBATCH --job-name=plot
#SBATCH -p n26
#SBATCH -o test.out
#SBATCH -e test.err
DIR='/home/fdebras/Multinest_from_calmip/Multinest_WASP76_LRHR/plot_post/'
source ~/.bashrc
cd $DIR
export OMP_NUM_THREADS=1
python3 plot.py
