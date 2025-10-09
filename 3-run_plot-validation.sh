#!/bin/bash
#SBATCH --job-name=plot_validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_44C_512G
#SBATCH --time=00:30:00
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta/log/%x/%x_%j.out
#SBATCH --error=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta/log/%x/%x_%j.err
# ------------------------------------------
# Plot cluster validation metrics from parallel clustering results
# Submit with: sbatch 3-run_plot-validation.sh
# Monitor with: squeue -u $USER

pwd; hostname; date
set -e

#==============Shell script==============#

PROJECT_DIR="/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta"
OUTPUT_DIR=${PROJECT_DIR}/derivatives/hierarchical_clustering

# Plotting parameters
K_MIN=2          # Minimum k value to plot
K_MAX=9          # Maximum k value to plot
FIGSIZE_W=12     # Figure width
FIGSIZE_H=5      # Figure height
DPI=300          # Figure resolution

echo "==================================================="
echo "CLUSTER VALIDATION PLOTTING"
echo "==================================================="
echo "Results directory: ${OUTPUT_DIR}"
echo "K range: ${K_MIN} to ${K_MAX}"
echo "Figure size: ${FIGSIZE_W}x${FIGSIZE_H}"
echo "DPI: ${DPI}"
echo "==================================================="

# Load environment
module load miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source activate /home/champ007/kmeans_env

# Run the plotting script
cmd="python -u ${PROJECT_DIR}/plot_cluster_validation.py \
    --results_dir ${OUTPUT_DIR} \
    --k_min ${K_MIN} --k_max ${K_MAX} \
    --figsize ${FIGSIZE_W} ${FIGSIZE_H} \
    --dpi ${DPI}"

echo "Commandline: $cmd"
eval $cmd

if [[ $? -eq 0 ]]; then
    echo "==================================================="
    echo "PLOTTING COMPLETED SUCCESSFULLY"
    echo "==================================================="
    echo "Plots saved to: ${OUTPUT_DIR}/figures/"
    echo "Files generated:"
    echo "  - cluster_validation_summary.png"
    echo "==================================================="
else
    echo "==================================================="
    echo "ERROR: Plotting failed!"
    echo "==================================================="
    exit 1
fi

exit 0

date
