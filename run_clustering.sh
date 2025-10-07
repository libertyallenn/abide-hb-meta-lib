#!/bin/bash
#SBATCH --job-name=clustering
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_44C_512G
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta-lib/log/%x/%x_%A-%a.out
#SBATCH --error=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta-lib/log/%x/%x_%A-%a.err
# ------------------------------------------
# SLURM Job Array for parallel clustering
# Array indices: 1=kmeans_no_pca, 2=kmeans_with_pca, 3=spectral

pwd; hostname; date
set -e

#==============Shell script==============#

PROJECT_DIR="/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta-lib"
OUTPUT_DIR=${PROJECT_DIR}/derivatives/clustering
DATA_DIR="/home/data/nbc/Laird_ABIDE/dset/derivatives/rsfc-habenula"

# Define clustering methods array
METHODS=("" "kmeans_no_pca" "kmeans_with_pca" "spectral" "kmeans_similarity")

# to run one of the clustering methods: 
#   ex) to run kmeans_similarity "sbatch --array=4 run_clustering.sh"

# Get the method for this array task
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "Running clustering method: $METHOD"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Load environment
module load miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source activate ${PROJECT_DIR}/kmeans_env 

# Setup done, run the command
cmd="python ${PROJECT_DIR}/clustering.py \
    --project_dir ${PROJECT_DIR} \
    --data_dir ${DATA_DIR} \
    --out_dir ${OUTPUT_DIR} \
    --method ${METHOD}"

echo "Commandline: $cmd"
eval $cmd

echo "Finished $METHOD clustering with exit code $?"

date