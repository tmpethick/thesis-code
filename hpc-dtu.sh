#!/bin/sh
#BSUB -q gputitanxpascal
#BSUB -gpu "num=1:mode=exclusive_process"

##BSUB -q gpuqueuek80
##BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -R "rusage[mem=8GB]"

#BSUB -J lions

#BSUB -n 1

#BSUB -W 00:10

#BSUB -o lions-%J.out
#BSUB -e lions-%J.err

#BSUB -L /bin/bash


# Prevents `plt.show()` for attempting connecting.
unset DISPLAY XAUTHORITY 

if test X$PBS_ENVIRONMENT = XPBS_BATCH; then cd $PBS_O_WORKDIR; fi


# Unload already installed software
module unload gcc
module unload cuda
module unload cudnn

# Setup cuda path
export CUDA_VERSION='9.0'
export CUDNN_VERSION='v7.4.2.24-prod-cuda-9.0'

# load modules
module load cuda/$CUDA_VERSION
module load cudnn/$CUDNN_VERSION
module load qt


cd ~/mthesis
source activate lions
echo "... Job beginning"
"$@"
echo "... Job Ended"
