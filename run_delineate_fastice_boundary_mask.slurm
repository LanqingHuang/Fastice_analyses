#!/bin/bash

#SBATCH --job-name=fastice_mask
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=fastice_mask_%j.out
#SBATCH --error=fastice_mask_%j.err
#SBATCH --chdir=/home/lhua0106/ce07/codes/FastIce_analyses

set -eo pipefail

ANALYSIS_DIR="/home/lhua0106/ce07/codes/FastIce_analyses"
PYTHON_SCRIPT="${ANALYSIS_DIR}/delineate_fast_ice_boundary_and_mask_auto.py"
CONDA_ENV="mylocal_coding"

# Use a writable Matplotlib cache on the compute node.
export MPLCONFIGDIR="/tmp/${USER}/matplotlib_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

# Limit numerical libraries to the CPUs assigned by Slurm.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

module load miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
echo "Started: $(date)"

# there are two mode: overwirte or skip if all six output PNGs exist
# Default behaviour: skip a pair when all six output PNGs already exist.
#python "${PYTHON_SCRIPT}"
# To force every eligible pair to be regenerated, replace the command above with:
python "${PYTHON_SCRIPT}" --overwrite

echo "Finished: $(date)"
