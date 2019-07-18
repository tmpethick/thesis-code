#!/bin/sh
# embedded options to bsub - start with #BSUB
### -- set the job Name --
#BSUB -J DP_500dGP
### –- specify queue --
#BSUB -q hpc
### -- ask for number of cores (default: 1) --
#BSUB -n 25
#BSUB -R "span[hosts=1]"
### -- set walltime limit: hh:mm --
#BSUB -W 48:00
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o DP_500dGP_%J.out
#BSUB -e DP_500dGP_%J.err

# load the necessary modules
# NOTE: this is just an example, check with the available modules
module load gcc
module load mpi/2.1.1-gcc-7.2.0

# Prevents `plt.show()` for attempting connecting.
unset DISPLAY XAUTHORITY 

cd ~/mthesis
source activate lions2
source setup_env.sh

export OMP_NUM_THREADS=1
### This uses the LSB_DJOB_NUMPROC to assign all the cores reserved
### This is a very basic syntax. For more complex examples, see the documentation
mpirun -np $LSB_DJOB_NUMPROC python run_growth_model.py
