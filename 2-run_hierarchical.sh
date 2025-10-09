#!/bin/bash
#SBATCH --job-name=hierarchical_clustering
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_44C_512G
#SBATCH --array=2-9  # Job array for k values (k=2 through k=9, total 8 jobs)
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta/log/%x/%x_%A-%a.out
#SBATCH --error=/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta/log/%x/%x_%A-%a.err
# ------------------------------------------
# Parallel Hierarchical Clustering - Each job runs one k value
# Submit with: sbatch 2-run_hierarchical.sh
# Monitor with: squeue -u $USER
# Results collected automatically when all jobs complete

pwd; hostname; date
set -e

#==============Shell script==============#

PROJECT_DIR="/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta"
OUTPUT_DIR=${PROJECT_DIR}/derivatives/hierarchical_clustering
DATA_DIR="/home/data/nbc/Laird_ABIDE/dset/derivatives/rsfc-habenula"

# Clustering parameters
K_MIN=2          # Minimum number of clusters to test
K_MAX=9          # Maximum number of clusters to test (8 total k values: 2,3,4,5,6,7,8,9)

# Get the k value for this job array task
K_VALUE=${SLURM_ARRAY_TASK_ID}

echo "==================================================="
echo "SLURM Job Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Running hierarchical analysis for k=${K_VALUE}"
echo "==================================================="

# Load environment
module load miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
source activate /home/champ007/kmeans_env

# Run the analysis for this k value (using preprocessed data)
cmd="python -u ${PROJECT_DIR}/hierarchical-workflow.py \
    --project_dir ${PROJECT_DIR} \
    --data_dir ${DATA_DIR} \
    --out_dir ${OUTPUT_DIR} \
    --k_value ${K_VALUE} \
    --load_preprocessed"

echo "Commandline: $cmd"
eval $cmd

echo "==================================================="
echo "Completed k=${K_VALUE} hierarchical analysis"
echo "==================================================="

# If this is the last job in the array, collect all results
if [[ ${SLURM_ARRAY_TASK_ID} == ${SLURM_ARRAY_TASK_MAX} ]]; then
    echo "This is the last job - waiting for all jobs to complete and collecting results..."
    
    # Wait a bit to ensure all jobs have finished writing their files
    sleep 30
    
    # Collect and summarize all results
    echo "Collecting results from all parallel jobs..."
    
    SUMMARY_FILE="${OUTPUT_DIR}/parallel_hierarchical_summary.txt"
    echo "Hierarchical Connectivity Clustering - Parallel Results Summary" > "${SUMMARY_FILE}"
    echo "Generated on: $(date)" >> "${SUMMARY_FILE}"
    echo "K range tested: ${K_MIN} to ${K_MAX}" >> "${SUMMARY_FILE}"
    echo "Method: hierarchical" >> "${SUMMARY_FILE}"
    echo "========================================" >> "${SUMMARY_FILE}"
    echo "" >> "${SUMMARY_FILE}"
    
    # Check for results and collect them
    BEST_SILHOUETTE=0
    BEST_K=0
    
    for k in $(seq ${K_MIN} ${K_MAX}); do
        RESULT_FILE="${OUTPUT_DIR}/k${k}_results.txt"
        if [[ -f "${RESULT_FILE}" ]]; then
            echo "Found results for k=${k}"
            echo "Results for k=${k}:" >> "${SUMMARY_FILE}"
            cat "${RESULT_FILE}" >> "${SUMMARY_FILE}"
            echo "" >> "${SUMMARY_FILE}"
            
            # Extract silhouette score for comparison
            SILHOUETTE=$(grep "Silhouette score:" "${RESULT_FILE}" | awk '{print $3}')
            if (( $(echo "${SILHOUETTE} > ${BEST_SILHOUETTE}" | bc -l) )); then
                BEST_SILHOUETTE=${SILHOUETTE}
                BEST_K=${k}
            fi
        else
            echo "WARNING: Results not found for k=${k}"
            echo "WARNING: Results not found for k=${k}" >> "${SUMMARY_FILE}"
        fi
    done
    
    # Add best k recommendation
    echo "" >> "${SUMMARY_FILE}"
    echo "========================================" >> "${SUMMARY_FILE}"
    echo "RECOMMENDATION:" >> "${SUMMARY_FILE}"
    if [[ ${BEST_K} -gt 0 ]]; then
        echo "Best k based on silhouette score: k=${BEST_K} (score: ${BEST_SILHOUETTE})" >> "${SUMMARY_FILE}"
        echo "Best k: ${BEST_K} (silhouette: ${BEST_SILHOUETTE})"
    else
        echo "No valid results found!" >> "${SUMMARY_FILE}"
        echo "WARNING: No valid results found!"
    fi
    
    
    echo "==================================================="
    echo "ALL HIERARCHICAL JOBS COMPLETED"
    echo "==================================================="
    echo "Summary file: ${SUMMARY_FILE}"
    echo "==================================================="
fi

exit 0

date
