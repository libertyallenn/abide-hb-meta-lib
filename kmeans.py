import os
import os.path as op
from glob import glob

import pandas as pd
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from nilearn.input_data import NiftiMasker
from nilearn.connectome import ConnectivityMeasure
from sklearn.cluster import KMeans

from utils import get_peaks, thresh_img

# Define input/output paths
project_dir = "/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta"
#metadata_file = op.join(project_dir, "derivatives/sub-group_task-rest_desc-1S2StTesthabenula_zmaps.txt") 
metadata_file = op.join(project_dir, "derivatives/sub-group_task-rest_desc-1S2StTesthabenula_conntabletest.txt")
base_rsfc_dir = "/home/data/nbc/Laird_ABIDE/dset/derivatives/rsfc-habenula"
output_dir = op.join(project_dir, "k_clustering", "hb_clustering_zmaps_asd")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------
# Step 1: Load and filter metadata
# -----------------------------------------
df = pd.read_csv(metadata_file, sep="\t", comment="#")
df_asd = df[df["group"] == "asd"].copy()

print(df_asd)

# Full z-map path for each ASD subject
#df_asd["zmap_path"] = df_asd["InputFile"].apply(lambda x: op.join(base_rsfc_dir, x.lstrip("/")))
df_asd["zmap_path"] = df_asd["InputFile"].str.replace(
    "^/rsfc/", "/home/data/nbc/Laird_ABIDE/dset/derivatives/rsfc-habenula/", regex=True
)


# Keep only entries where the zmap file exists
#df_asd = df_asd[df_asd["zmap_path"].apply(op.exists)]

print(df_asd)

# Create list of z-map paths and subject IDs
z_maps = df_asd["zmap_path"].tolist()
subject_ids = df_asd["Subj"].tolist()

print(f"Found {len(z_maps)} ASD subjects with valid z-maps.")

# -----------------------------------------
# Step 2: Load and prepare data
# -----------------------------------------
masker = NiftiMasker(mask_strategy="background", standardize=True)
data_matrix = masker.fit_transform(z_maps)  # shape: (n_subjects, n_voxels)

# Compute correlation matrix (subject similarity)
correlation = ConnectivityMeasure(kind="correlation")
corrmat = correlation.fit_transform([data_matrix])[0]

# -----------------------------------------
# Step 3: Run KMeans clustering
# -----------------------------------------
for d in range(2, 7):  # cluster solutions: k=2 to k=6
    clustering = KMeans(n_clusters=d, n_init=100, max_iter=1000, random_state=0)
    cluster_labels = clustering.fit(1 - corrmat).labels_

    tmp_output_dir = op.join(output_dir, f"cluster_{d}")
    os.makedirs(tmp_output_dir, exist_ok=True)

    for a in range(d):
        cluster_indices = np.where(cluster_labels == a)[0]
        cluster_zs = [z_maps[i] for i in cluster_indices]

        # Average z-map for this cluster
        cluster_data = masker.transform(cluster_zs)
        cluster_mean = np.mean(cluster_data, axis=0)
        cluster_mean_img = masker.inverse_transform(cluster_mean)

        # Save average z-map
        mean_img_path = op.join(tmp_output_dir, f"cluster_{a}_mean_zmap.nii.gz")
        cluster_mean_img.to_filename(mean_img_path)

        # Threshold and save
        z_img_thresh = thresh_img(cluster_mean_img, cluster_mean_img, 0.001)
        z_thresh_path = op.join(tmp_output_dir, f"cluster_{a}_z_thresh-001.nii.gz")
        nib.save(z_img_thresh, z_thresh_path)

        # Extract peaks
        get_peaks(z_thresh_path, tmp_output_dir)

        # Save subject list for this cluster
        with open(op.join(tmp_output_dir, f"cluster_{a}_subject_ids.txt"), "w") as f:
            for idx in cluster_indices:
                f.write(subject_ids[idx] + "\n")
