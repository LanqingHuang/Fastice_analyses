#!/bin/bash

#SBATCH --job-name=insar_timeseries
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=insar_timeseries_%j.out
#SBATCH --error=insar_timeseries_%j.err
#SBATCH --chdir=/home/lhua0106/ce07/codes/FastIce_analyses

set -eo pipefail

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mylocal_coding

# Prevent Matplotlib from trying to open a graphical window
export MPLBACKEND=Agg

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Python: $(which python)"

python -u analyses_time_series_auto.py

echo "Job completed: $(date)"