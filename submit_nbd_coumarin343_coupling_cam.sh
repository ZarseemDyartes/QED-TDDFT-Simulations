#!/bin/bash
#SBATCH --job-name=nbd_c343
#SBATCH --account=mhg
#SBATCH --partition=mhg
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=16G                   
#SBATCH --time=32:00:00            
#SBATCH --output=slurm-%j.log      

# 1. Load the Conda module
module load miniconda3/22.11.1-gcc-11.4.0

# 2. Source the conda setup script
source /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-11.4.0/miniconda3-22.11.1-bfe6srpvwx7su5dkxaeljoj3jmxnww43/etc/profile.d/conda.sh

# 3. Activate your environment
conda activate pyscf_env

# ========================================================
# 4. OFFICIAL GITHUB FIX: Tell Python where the QED module is
# ========================================================
export PYTHONPATH="/global/scratch/users/vaibhavvaiyakarnam/bdp_tddft/qed-tddft:$PYTHONPATH"

# 5. Set threading variables to match requested CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 6. Run the optimized QED Lambda Scan script
python nbd_coumarin343_coupling_cam.py > nbd_coumarin343_coupling_cam.out