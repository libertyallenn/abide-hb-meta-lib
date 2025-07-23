import os
import os.path as op
from glob import glob

import pandas as pd
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from nilearn.maskers import NiftiMasker
from nilearn.connectome import ConnectivityMeasure
from sklearn.cluster import KMeans

from utils import get_peaks, thresh_img

# Define input/output paths
project_dir = "/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta"
#metadata_file = op.join(project_dir, "derivatives/sub-group_task-rest_desc-1S2StTesthabenula_zmaps.txt") 
metadata_file = op.join(project_dir, "derivatives/sub-group_task-rest_desc-1S2StTesthabenula_conntable.txt")
base_rsfc_dir = "/home/data/nbc/Laird_ABIDE/dset/derivatives/rsfc-habenula"
output_dir = op.join(project_dir, "derivatives", "k_clustering")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------
# Step 1: Load and filter metadata
# -----------------------------------------
df = pd.read_csv(metadata_file, sep="\t", comment="#")
df_asd = df[df["group"] == "asd"].copy()

print(df_asd)

# Full z-map path for each ASD subject
#df_asd["zmap_path"] = df_asd["InputFile"].apply(lambda x: op.join(base_rsfc_dir, x.lstrip("/")))
df_asd["bmap_path"] = df_asd["InputFile"].str.replace(
    "^/rsfc/", "/home/data/nbc/Laird_ABIDE/dset/derivatives/rsfc-habenula/", regex=True
)


# Keep only entries where the zmap file exists
#df_asd = df_asd[df_asd["zmap_path"].apply(op.exists)]

print(df_asd)

# Create list of connectivity map paths and subject IDs
b_maps = df_asd["bmap_path"].tolist()
subject_ids = df_asd["Subj"].tolist()

print(f"Found {len(b_maps)} ASD subjects with valid beta maps.")

# Optional: confirm NIfTI loading works
for path in b_maps:
    try:
        nib.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")

# -----------------------------------------
# Step 2: Load and prepare data
# -----------------------------------------
masker = NiftiMasker(mask_strategy="background", standardize=False)
data_matrix = masker.fit_transform(b_maps)  # shape: (n_subjects, n_voxels)

# Compute correlation matrix (subject similarity)
corrmat = np.corrcoef(data_matrix)

# correlation = ConnectivityMeasure(kind="correlation")
# corrmat = correlation.fit_transform([data_matrix])[0]

# -----------------------------------------
# Step 3: Run KMeans clustering
# -----------------------------------------
for d in range(2, 9):  
    clustering = KMeans(n_clusters=d, n_init=100, max_iter=1000, random_state=0)
    cluster_labels = clustering.fit(1 - corrmat).labels_

    tmp_output_dir = op.join(output_dir, f"cluster_{d}")
    os.makedirs(tmp_output_dir, exist_ok=True)

    for a in range(d):
        cluster_indices = np.where(cluster_labels == a)[0]
        cluster_maps = [b_maps[i] for i in cluster_indices]

        # Average beta maps for this cluster
        cluster_data = masker.transform(cluster_maps)
        cluster_mean = np.mean(cluster_data, axis=0)
        cluster_mean_img = masker.inverse_transform(cluster_mean)

        # Save mean beta map
        beta_map_path = op.join(tmp_output_dir, f"cluster_{a}_mean_betamap.nii.gz")
        nib.save(cluster_mean_img, beta_map_path)

        # Threshold beta map (set threshold value appropriate for beta maps, e.g., 0.2 or 0.3)
        beta_thresh_value = 0.2  # adjust as needed
        beta_img_thresh = thresh_img(cluster_mean_img, cluster_mean_img, beta_thresh_value)
        beta_thresh_path = op.join(tmp_output_dir, f"cluster_{a}_beta_thresh-{beta_thresh_value}.nii.gz")
        nib.save(beta_img_thresh, beta_thresh_path)

        # Extract peaks
        get_peaks(beta_thresh_path, tmp_output_dir)

        # ---------------------------
        # Create and save Z-score map
        # ---------------------------
        cluster_mean_data = cluster_mean_img.get_fdata()
        mean_val = np.mean(cluster_mean_data)
        std_val = np.std(cluster_mean_data)

        if std_val == 0:
            print(f"Warning: Cluster {a} mean beta map has zero std, skipping z-transform.")
            continue

        z_map_data = (cluster_mean_data - mean_val) / std_val
        z_img = nib.Nifti1Image(z_map_data, cluster_mean_img.affine, cluster_mean_img.header)

        # Save unthresholded Z-map
        z_map_path = op.join(tmp_output_dir, f"cluster_{a}_mean_zmap.nii.gz")
        nib.save(z_img, z_map_path)

        # Threshold Z-map (z > 3.09 corresponds to p < 0.001)
        z_thresh_value = 3.09
        z_img_thresh = thresh_img(z_img, z_img, z_thresh_value)
        z_thresh_path = op.join(tmp_output_dir, f"cluster_{a}_z_thresh-{z_thresh_value}.nii.gz")
        nib.save(z_img_thresh, z_thresh_path)

        # Extract peaks for z-thresholded map
        get_peaks(z_thresh_path, tmp_output_dir)

        # Save subject list for this cluster
        with open(op.join(tmp_output_dir, f"cluster_{a}_subject_ids.txt"), "w") as f:
            for idx in cluster_indices:
                f.write(subject_ids[idx] + "\n")

        # Print completion for individual cluster
        print(f"Completed cluster {a} for k = {d}")

    # Print completion for the entire k-cluster solution
    print(f"Completed all {d} clusters (k = {d})\n")