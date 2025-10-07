# abide-hb-meta

Clustering autistic participants based on **whole-brain habenula functional connectivity** (rsFC).  
This project builds participant × participant similarity matrices from rsFC maps and applies K-means clustering across a range of *k*, saving cluster assignments, centroid maps, and QC figures.

---

# abide-hb-meta

Clustering autistic participants based on whole-brain habenula functional connectivity (rsFC). The repository contains several workflows and convenience scripts to build participant-by-participant similarity matrices, run K-means clustering across a range of k, and save cluster assignments, centroid maps, and QC figures.

---

## Files of interest

Top-level scripts and important files in this directory:

- `kmeans-workflow.py` — Main end-to-end pipeline: loads maps, builds similarity matrices, runs K-means across a range of k, and writes cluster outputs and figures.
- `kmeans_similarity_workflow.py` — Focused workflow that builds participant × participant similarity matrices and related QC visualizations. Use this when you only need the similarity matrix and diagnostics.
- `clustering-workflow.py` — Alternative clustering driver (a smaller/experimental workflow that can be used to apply clustering using precomputed similarity/distance matrices).
- `utils.py` — Helper functions used by the workflows (masking, thresholding, I/O helpers).
- `kmeans_env.yml` — Conda environment specification used to reproduce the Python environment used during development.
- `run_kmeans.sh` — Shell wrapper / HPC launcher for the `kmeans-workflow.py` (may be used with `sbatch` on SLURM clusters or run locally).
- `run_kmeans_similarity.sh` — Wrapper script to run `kmeans_similarity_workflow.py` (HPC/launcher convenience script).
- `run_clustering.sh` — Wrapper script to run `clustering-workflow.py`.
- `derivatives/` — Output directory (contains example output files in this repo).

There are example derivative files in `derivatives/` included for reference (e.g., `sub-group_task-rest_desc-1S2StTesthabenula_conntable*.txt`).

---

## Quick overview of the pipeline steps

Typical processing steps implemented across the workflows:

1. Load metadata (TSV/TXT) and construct paths to participant rsFC maps.
2. Mask & vectorize maps into a participant × voxel data matrix.
3. Compute similarity across participants (Pearson correlation by default).
4. Optionally reduce dimensionality (PCA) and evaluate cluster validity (silhouette, elbow) to help select k.
5. Perform clustering (K-means) across a user-specified range of k.
6. Generate outputs: centroid beta maps, z-maps, thresholded maps, cluster assignment lists, and QC figures (similarity heatmaps, validation plots, PCA scatter plots, etc.).

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

Workflows write outputs under `derivatives/` (for example `derivatives/k_clustering_matrix/`), including:

- `figures/` — diagnostic plots (similarity heatmaps, cluster validation plots, cluster visualizations)
- `k_{k}_clusters/` — per-k folders containing cluster centroid maps and participant lists
- `cluster_assignments_k_{k}.csv` — table of cluster assignments for each subject at a given k

Exact filenames and folder layout vary slightly between workflows, but all workflows include cluster assignments, centroid maps, and QC figures.

---

## Software / Environment

This project was developed with Python 3.9+ and common scientific packages. A conda environment spec is provided as `kmeans_env.yml` — create it with:

```bash
conda env create -f kmeans_env.yml
conda activate <env-name-from-yml>
```

If you prefer pip, ensure you have the usual packages: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `nibabel`, `nilearn` (and `fastcluster` if required by any particular script).

---

## Usage examples

Run the main K-means workflow locally (edit configuration at the top / `main()` of the script):

```bash
python kmeans-workflow.py
```

Build only the similarity matrix and QC outputs:

```bash
python kmeans_similarity_workflow.py
```

Run the alternative clustering driver:

```bash
python clustering-workflow.py
```

Wrapper scripts (convenience/HPC):

```bash
# run locally or on a compute node
bash run_kmeans.sh
bash run_kmeans_similarity.sh
bash run_clustering.sh

# or submit to SLURM if the script is configured for sbatch
sbatch run_kmeans.sh
sbatch run_kmeans_similarity.sh
sbatch run_clustering.sh
```

If you update scripts or need a different metadata path, edit the `main()` section in the relevant Python file(s) to point to your `project_dir`, `metadata_file`, and `base_rsfc_dir`.

---

## Troubleshooting

- If imports fail, ensure you activated the conda environment created from `kmeans_env.yml`.
- Check that the metadata `InputFile` paths point to existing NIfTI files (or update `base_rsfc_dir` in the scripts).
- Inspect stdout/logs produced by the `run_*.sh` scripts for runtime errors (they may include SLURM directives).

---

## License
[MIT License](LICENSE)
