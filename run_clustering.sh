#!/bin/bash
#SBATCH --job-name=kmeans
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_40C_512G
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta/log/%x/%x_%A-%a.out
#SBATCH --error=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta/log/%x/%x_%A-%a.err
# ------------------------------------------
# Max # CPUs = 360
# modified from ABIDE workflow

# IB_44C_512G, IB_40C_512G, IB_16C_96G, for running workflow
# investor, for testing
pwd; hostname; date
set -e

#==============Shell script==============#
# Load evironment
module load miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source activate /home/champ007/kmeans_env

PROJECT_DIR="/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta"

# Setup done, run the command
cmd="python ${PROJECT_DIR}/kmeans.py \
    --project_dir ${PROJECT_DIR} \
    --n_cores ${SLURM_CPUS_PER_TASK}"
echo Commandline: $cmd
eval $cmd


#echo Finished tasks with exit code $exitcode
exit $exitcode

date