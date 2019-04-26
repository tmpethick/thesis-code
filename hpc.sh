#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4      # cores requested
#SBATCH --mem=1024  # memory in Mb
##SBATCH -o log.out  # send stdout to outfile
##SBATCH -e log.err  # send stderr to errfile
#SBATCH -t 0:01:00  # time requested in hour:minute:second
cd ~/mthesis
source activate lions
echo "... Job beginning"
"$@"
echo "... Job Ended"
