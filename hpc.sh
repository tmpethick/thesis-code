#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4      # cores requested
#SBATCH --mem=8192  # memory in Mb
##SBATCH -o log.out  # send stdout to outfile
##SBATCH -e log.err  # send stderr to errfile
#SBATCH -t 10:00:00  # time requested in hour:minute:second

# Prevents `plt.show()` for attempting connecting.
unset DISPLAY XAUTHORITY 

cd ~/mthesis
source activate lions
echo "... Job beginning"
"$@"
echo "... Job Ended"
