# abide-hb-meta

**Hierarchical clustering of autistic participants based on whole-brain habenula functional connectivity patterns.**

This project identifies subgroups of autistic participants who share similar habenula connectivity patterns across the brain. We use hierarchical clustering to discover natural groupings in the data and employ silhouette scores and gap statistics as stability metrics to determine the optimal number of clusters.

---

## Overview

We performed hierarchical clustering analysis to identify similarities between autistic participants in their habenula connectivity patterns. The approach:

1. **Data preparation**: Whole-brain habenula functional connectivity (rsFC) maps from autistic participants
2. **Hierarchical clustering**: Ward linkage clustering on correlation-distance to identify natural groupings
3. **Cluster validation**: Silhouette analysis and gap statistics to determine optimal cluster solutions
4. **Visualization**: Comprehensive plots and validation metrics to aid interpretation

### Key Features

- **Hierarchical clustering**: Discovers natural groupings without pre-specifying cluster count
- **Multiple validation metrics**: Both silhouette scores and gap statistics for robust cluster evaluation
- **Parallel processing**: Efficient computation across different k values (k=2 to k=8)
- **Comprehensive outputs**: Cluster assignments, validation plots, and summary statistics

---

## Choosing the Best Cluster Solution

We use two complementary stability metrics to determine the optimal number of clusters:

### 1. Silhouette Analysis
- **What it measures**: How well each participant fits within their assigned cluster vs. other clusters
- **Range**: -1 to +1 (higher is better)
- **Interpretation**: 
  - Values > 0.5: Strong cluster structure
  - Values 0.2-0.5: Reasonable structure
  - Values < 0.2: Weak or artificial structure

### 2. Gap Statistic
- **What it measures**: Compares within-cluster dispersion to expected dispersion under null hypothesis
- **Interpretation**: Higher values indicate better clustering
- **Tibshirani rule**: Choose smallest k where Gap(k) ≥ Gap(k+1) - s_{k+1}

### How to Choose:
1. **Look for silhouette score peaks**: The k value with the highest silhouette score often indicates optimal clustering
2. **Apply gap statistic rule**: Use Tibshirani rule for statistical rigor
3. **Consider interpretability**: Balance statistical metrics with biological/clinical interpretability
4. **Check cluster sizes**: Ensure clusters have reasonable sample sizes for meaningful analysis

The validation plots show both metrics across k=2 to k=8, with clear annotations highlighting the optimal solutions.

---

## Files of interest

Top-level scripts and important files in this directory:

- `1-run_data-matrix.sh` / `run_data-matrix.sh` — Preprocessing wrapper to create the data matrix (numbered: 1-run_data-matrix.sh).
- `2-run_hierarchical.sh` / `run_hierarchical.sh` — Shell wrapper for running `hierarchical-workflow.py` on HPC clusters or locally (numbered: 2-run_hierarchical.sh).
- `3-run_plot-validation.sh` / `run_plot-validation.sh` — Script to generate cluster validation plots (numbered: 3-run_plot-validation.sh).
- `run_kmeans.sh` — Optional wrapper to run the K-means workflow (`kmeans-workflow.py`) if you prefer that method.
- `hierarchical-workflow.py` — Main hierarchical clustering pipeline: loads connectivity maps, performs Ward hierarchical clustering, calculates validation metrics, and generates comprehensive outputs.
- `plot_cluster_validation.py` — Creates validation plots showing silhouette scores and gap statistics across k values to help determine optimal cluster solutions.
- `utils.py` — Helper functions used by the workflows (masking, thresholding, I/O helpers).
- `kmeans_env.yml` — Conda environment specification for reproducing the analysis environment.
- `derivatives/` — Output directory containing results:
  - `hierarchical_clustering/` — Main results directory
    - `k_2/` to `k_8/` — Individual cluster solutions
    - `figures/` — Validation plots and dendrograms
    - `cluster_validation_metrics.csv` — Combined validation metrics DataFrame

### Optional / alternative workflows
- `kmeans-workflow.py` — Optional K-means clustering workflow. This is provided as an alternative to the hierarchical pipeline; use `run_kmeans.sh` to run it. It is not part of the numbered wrapper ordering by default.

## Repository structure (quick file map)

Top-level scripts you'll likely use (numbered wrappers are recommended):

- `1-run_data-matrix.sh` — Create the preprocessed data matrix (preprocessing; run once).
- `2-run_hierarchical.sh` — Parallel hierarchical clustering wrapper (recommended for the main pipeline).
- `3-run_plot-validation.sh` — Generate cluster validation plots from the hierarchical results.
- `run_kmeans.sh` — Optional: run the K-means workflow (`kmeans-workflow.py`) as a replacement to hierarchical clustering if you prefer that method.

Notes:
- Numbered wrappers (1,2,3) are provided for a clear execution order; the original unnumbered scripts are still present for backwards compatibility but the README and workflow now recommend the numbered names.
- The K-means workflow is optional — if you prefer k-means clustering instead of the hierarchical approach, run `run_kmeans.sh` / `kmeans-workflow.py` and inspect the outputs under `derivatives/k_clustering/`.

---

## Pipeline Steps

The hierarchical clustering pipeline follows these steps:

1. **Data loading**: Load metadata and validate habenula rsFC map paths
2. **Data matrix creation**: Mask and vectorize connectivity maps into participant × voxel matrix
3. **Hierarchical clustering**: Apply Ward linkage clustering across k=2 to k=8
4. **Validation metrics**: Calculate silhouette scores and gap statistics for each k
5. **Output generation**: Create cluster assignments, validation plots, and summary statistics
6. **Visualization**: Generate dendrograms, PCA plots, and validation summaries

---

## Inputs

- A metadata file (TSV/TXT) with at least the following columns (names can vary slightly depending on which workflow you use — check the script headers):
  - `Subj` → subject ID
  - `group` → group label (e.g., `asd`)
  - `InputFile` → path to each participant's rsFC NIfTI (or a relative path that can be rewritten via `base_rsfc_dir`)
- A directory containing rsFC maps referenced by the metadata file (set `base_rsfc_dir` or edit the script's `main()` configuration section).

Note: some workflows default to filtering rows by `group == "asd"`. You can modify that behavior in the script or supply a metadata file pre-filtered to your group of interest.

---

## Outputs

The hierarchical clustering workflow generates comprehensive outputs under `derivatives/hierarchical_clustering/`:

### Main Results
- `cluster_validation_metrics.csv` — Combined DataFrame with silhouette scores, gap statistics, and validation metrics for all k values
- `hierarchical_cluster_validation.txt` — Summary of validation results and recommendations

### Per-k Results (k_2/ through k_8/)
- `hierarchical_connectivity_groups_k{k}.csv` — Cluster assignments for each participant
- `k{k}_results.txt` — Detailed validation metrics for this k value
- `figures/` — Visualization plots for this cluster solution

### Validation Plots
- `figures/cluster_validation_summary.png` — Combined silhouette and gap statistic plots
- `figures/hierarchical_dendrogram.png` — Hierarchical clustering dendrogram
- `figures/hierarchical_connectivity_groups_k{k}.png` — PCA visualization and cluster size plots

### Key Output Files
- **Cluster assignments**: CSV files showing which cluster each participant belongs to
- **Validation metrics**: Quantitative measures to determine optimal k
- **Visualization plots**: Clear graphics showing cluster structure and validation results

---

## Software / Environment

This project was developed with Python 3.9+ and common scientific packages. A conda environment spec is provided as `kmeans_env.yml` — create it with:

```bash
conda env create -f kmeans_env.yml
conda activate <env-name-from-yml>
```

If you prefer pip, ensure you have the usual packages: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `nibabel`, `nilearn` (and `fastcluster` if required by any particular script).

---

## Usage Examples

### Full Hierarchical Clustering Analysis

Run the complete hierarchical clustering workflow:

```bash
# Full analysis (k=2 to k=8)
python hierarchical-workflow.py \
    --project_dir /path/to/project \
    --data_dir /path/to/rsfc/data \
    --out_dir derivatives/hierarchical_clustering \
    --k_min 2 --k_max 8
```

### Parallel Processing (Recommended for HPC)

```bash
# Step 1: Preprocess data matrix (run once)
python hierarchical-workflow.py \
    --project_dir /path/to/project \
    --data_dir /path/to/rsfc/data \
    --out_dir derivatives/hierarchical_clustering \
    --preprocess_only

# Step 2: Submit parallel jobs for each k value
for k in {2..8}; do
  sbatch --export=K_VALUE=$k 2-run_hierarchical.sh
done
```

### Generate Validation Plots

Create validation plots from results:

```bash
# Generate cluster validation plots
python plot_cluster_validation.py \
    --results_dir derivatives/hierarchical_clustering \
    --k_min 2 --k_max 8
```

### Wrapper Scripts (HPC/Local)

```bash
# Run complete workflow locally (numbered wrappers)
bash 2-run_hierarchical.sh

# Generate validation plots
bash 3-run_plot-validation.sh

# Submit to SLURM cluster using numbered wrapper
sbatch 2-run_hierarchical.sh
```

### Using the Results

After running the analysis:

1. **Check validation plots**: Look at `figures/cluster_validation_summary.png` to see silhouette and gap statistics
2. **Review recommendations**: Check `hierarchical_cluster_validation.txt` for suggested k values
3. **Examine cluster assignments**: Use `hierarchical_connectivity_groups_k{k}.csv` files for downstream analysis
4. **Inspect individual solutions**: Review per-k results in `k_{k}/` directories

---

## Troubleshooting

- **Import errors**: Ensure you activated the conda environment created from `kmeans_env.yml`
- **Missing rsFC files**: Check that metadata `InputFile` paths point to existing NIfTI files or update `base_rsfc_dir`
- **Parallel job issues**: Verify SLURM directives in wrapper scripts match your cluster configuration
- **Memory issues**: Large datasets may require more memory; adjust job parameters or use preprocessing mode
- **Plotting errors**: Ensure the results directory contains either `cluster_validation_metrics.csv` or individual `k*_results.txt` files
- **Empty clusters**: If clusters are too small, consider using a smaller k range or different clustering parameters

### Common Issues

- **Script hangs on visualization**: The plotting functions now use `plt.close()` instead of `plt.show()` for headless environments
- **File not found errors**: Check that the hierarchical clustering results exist before running validation plots
- **Inconsistent results**: Ensure all parallel jobs complete before generating summary plots

---

## Citation

If you use this clustering approach, please cite the relevant methodological papers for hierarchical clustering, silhouette analysis, and gap statistics.

---

## License
[MIT License](LICENSE)
