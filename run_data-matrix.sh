#!/bin/bash
#SBATCH --job-name=data-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_44C_512G
#SBATCH --time=02:00:00
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta/log/%x/%x_%j.out
#SBATCH --error=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta/log/%x/%x_%j.err
# ------------------------------------------
# Preprocessing step: Create data matrix once
# Submit with: sbatch run_data-matrix.sh

pwd; hostname; date
set -e

PROJECT_DIR="/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta"
OUTPUT_DIR=${PROJECT_DIR}/derivatives/hierarchical_clustering
DATA_DIR="/home/data/nbc/Laird_ABIDE/dset/derivatives/rsfc-habenula"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}
echo "Created/verified output directory: ${OUTPUT_DIR}"

echo "==================================================="
echo "PREPROCESSING: Creating data matrix for clustering"
echo "==================================================="

# Load environment
module load miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source activate /home/champ007/kmeans_env

# Run preprocessing only
cmd="python -u ${PROJECT_DIR}/hierarchical-workflow.py \
    --project_dir ${PROJECT_DIR} \
    --data_dir ${DATA_DIR} \
    --out_dir ${OUTPUT_DIR} \
    --preprocess_only"

echo "Commandline: $cmd"
eval $cmd

echo "==================================================="
echo "PREPROCESSING COMPLETED"
echo "Data matrix saved to: ${OUTPUT_DIR}/clustering_data_matrix.npy"
echo "Metadata saved to: ${OUTPUT_DIR}/clustering_metadata.json"
echo "Now submit parallel hierarchical clustering: sbatch run_hierarchical.sh"
echo "==================================================="

date