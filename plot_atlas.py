import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import image, plotting

# Paths to the Nifti files for each of the clusters
nifti_file10 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_1/cluster_0_z_corr-cFWE_thresh-001.nii.gz"

"""nifti_file20 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_2/cluster_0_z_corr-cFWE_thresh-001.nii.gz"
nifti_file21 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_2/cluster_1_z_corr-cFWE_thresh-001.nii.gz"

nifti_file30 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_3/cluster_0_z_corr-cFWE_thresh-001.nii.gz"
nifti_file31 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_3/cluster_1_z_corr-cFWE_thresh-001.nii.gz"
nifti_file32 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_3/cluster_2_z_corr-cFWE_thresh-001.nii.gz"

nifti_file40 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_4/cluster_0_z_corr-cFWE_thresh-001.nii.gz"
nifti_file41 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_4/cluster_1_z_corr-cFWE_thresh-001.nii.gz"
nifti_file42 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_4/cluster_2_z_corr-cFWE_thresh-001.nii.gz"

nifti_file50 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_5/cluster_0_z_corr-cFWE_thresh-001.nii.gz"
nifti_file51 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_5/cluster_1_z_corr-cFWE_thresh-001.nii.gz"
nifti_file52 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_5/cluster_2_z_corr-cFWE_thresh-001.nii.gz"
nifti_file53 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_5/cluster_3_z_corr-cFWE_thresh-001.nii.gz"

nifti_file61 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_6/cluster_1_z_corr-cFWE_thresh-001.nii.gz"
nifti_file62 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_6/cluster_2_z_corr-cFWE_thresh-001.nii.gz"
nifti_file63 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_6/cluster_3_z_corr-cFWE_thresh-001.nii.gz"
nifti_file65 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_6/cluster_5_z_corr-cFWE_thresh-001.nii.gz"

nifti_file71 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_7/cluster_1_z_corr-cFWE_thresh-001.nii.gz"
nifti_file72 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_7/cluster_2_z_corr-cFWE_thresh-001.nii.gz"
nifti_file74 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_7/cluster_4_z_corr-cFWE_thresh-001.nii.gz"
nifti_file75 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_7/cluster_5_z_corr-cFWE_thresh-001.nii.gz"
nifti_file76 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_7/cluster_6_z_corr-cFWE_thresh-001.nii.gz"

nifti_file83 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_8/cluster_3_z_corr-cFWE_thresh-001.nii.gz"
nifti_file84 = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_8/cluster_4_z_corr-cFWE_thresh-001.nii.gz"
nifti_file85 = /home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis/k_clustering/clustering_856/cluster_8/cluster_5_z_corr-cFWE_thresh-001.nii.gz"""

# Load the two Nifti files for each of the clusters
nifit_img10 = image.load_img(nifti_file10)

"""nifti_img20 = image.load_img(nifti_file20)
nifti_img21 = image.load_img(nifti_file21)

nifti_img30 = image.load_img(nifti_file30)
nifti_img31 = image.load_img(nifti_file31)
nifti_img32 = image.load_img(nifti_file32)

nifti_img40 = image.load_img(nifti_file40)
nifti_img41 = image.load_img(nifti_file41)
nifti_img42 = image.load_img(nifti_file42)

nifti_img50 = image.load_img(nifti_file50)
nifti_img51 = image.load_img(nifti_file51)
nifti_img52 = image.load_img(nifti_file52)
nifti_img53 = image.load_img(nifti_file53)

nifti_img61 = image.load_img(nifti_file61)
nifti_img62 = image.load_img(nifti_file62)
nifti_img63 = image.load_img(nifti_file63)
nifti_img65 = image.load_img(nifti_file65)

nifti_img71 = image.load_img(nifti_file71)
nifti_img72 = image.load_img(nifti_file72)
nifti_img74 = image.load_img(nifti_file74)
nifti_img75 = image.load_img(nifti_file75)
nifti_img76 = image.load_img(nifti_file76)

nifti_img83 = image.load_img(nifti_file83)
nifti_img84 = image.load_img(nifti_file84)
nifti_img85 = image.load_img(nifti_file85)"""

# create a color map for the ROIs
roi_colors = ["fuchsia", "fuchsia"]
cmap = ListedColormap(roi_colors)

# Create a figure and axis for k=1 solutions
fig, ax = plt.subplots()

overlay_img = image.math_img("img10", img10=nifit_img10)

plotting.plot_roi(
    overlay_img, cut_coords=(50, -57, 4), cmap=cmap, alpha=0.5, colorbar=True, axes=ax
)

# Save the figure
output_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"
output_file = os.path.join(output_dir, "atlas1.png")
fig.savefig(output_file)

print(f"Figure saved as {output_file}")

"""# create a color map for the ROIs
roi_colors = ["fuchsia", "teal"]
cmap = ListedColormap(roi_colors)

# Create a figure and axis for k=2 solutions
fig, ax = plt.subplots()

overlay_img = image.math_img("img20 + img21", img20=nifti_img20, img21=nifti_img21)

plotting.plot_roi(
    overlay_img, cut_coords=(50, -57, 4), cmap=cmap, alpha=0.5, colorbar=True, axes=ax
)

# Save the figure
output_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"
output_file = os.path.join(output_dir, "atlas2.png")
fig.savefig(output_file)

print(f"Figure saved as {output_file}")

roi_colors = ["fuchsia","teal", "lime"]
cmap = ListedColormap(roi_colors)

# Create a figure and axis for k=3 solutions
fig, ax = plt.subplots()

overlay_img = image.math_img(
    "img30 + img31 + img32", img30=nifti_img30, img31=nifti_img31, img32=nifti_img32
)

plotting.plot_roi(
    overlay_img, cut_coords=(50, -57, 4), cmap=cmap, alpha=0.5, colorbar=True, axes=ax
)

# Save the figure
output_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"
output_file = os.path.join(output_dir, "atlas3.png")
fig.savefig(output_file)

print(f"Figure saved as {output_file}")

roi_colors = ["teal", "fuchsia", "lime"]
cmap = ListedColormap(roi_colors)


# Create a figure and axis for k=4 solutions
fig, ax = plt.subplots()

overlay_img = image.math_img(
    "img40 + img41 + img42", img40=nifti_img40, img41=nifti_img41, img42=nifti_img42
)

plotting.plot_roi(
    overlay_img, cut_coords=(50, -57, 4), cmap=cmap, alpha=0.5, colorbar=True, axes=ax
)

# Save the figure
output_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"
output_file = os.path.join(output_dir, "atlas4.png")
fig.savefig(output_file)

print(f"Figure saved as {output_file}")

roi_colors = ["teal", "fuchsia", "lime", "orange"]
cmap = ListedColormap(roi_colors)


# Create a figure and axis for k=5 solutions
fig, ax = plt.subplots()

overlay_img = image.math_img(
    "img50 + img51 + img52 + img53",
    img50=nifti_img50,
    img51=nifti_img51,
    img52=nifti_img52,
    img53=nifti_img53,
)

plotting.plot_roi(
    overlay_img, cut_coords=(50, -57, 4), cmap=cmap, alpha=0.5, colorbar=True, axes=ax
)

# Save the figure
output_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"
output_file = os.path.join(output_dir, "atlas5.png")
fig.savefig(output_file)

print(f"Figure saved as {output_file}")

roi_colors = ["teal", "fuchsia", "lime", "orange"]
cmap = ListedColormap(roi_colors)

# Create a figure and axis for k=5 solutions
fig, ax = plt.subplots()

overlay_img = image.math_img(
    "img61 + img62 + img63 + img65",
    img61=nifti_img61,
    img62=nifti_img62,
    img63=nifti_img63,
    img65=nifti_img65,
)

plotting.plot_roi(
    overlay_img, cut_coords=(50, -57, 4), cmap=cmap, alpha=0.5, colorbar=True, axes=ax
)

# Save the figure
output_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"
output_file = os.path.join(output_dir, "atlas6.png")
fig.savefig(output_file)

print(f"Figure saved as {output_file}")

roi_colors = ["teal", "fuchsia", "lime", "orange", "indigo"]
cmap = ListedColormap(roi_colors)

# Create a figure and axis for k=5 solutions
fig, ax = plt.subplots()

overlay_img = image.math_img(
    "img71 + img72 + img74 + img75 + img76",
    img71=nifti_img71,
    img72=nifti_img72,
    img74=nifti_img74,
    img75=nifti_img75,
    img76=nifti_img76,
)

plotting.plot_roi(
    overlay_img, cut_coords=(50, -57, 4), cmap=cmap, alpha=0.5, colorbar=True, axes=ax
)

# Save the figure
output_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"
output_file = os.path.join(output_dir, "atlas7.png")
fig.savefig(output_file)

print(f"Figure saved as {output_file}")

roi_colors = ["teal", "fuchsia", "lime"]
cmap = ListedColormap(roi_colors)

# Create a figure and axis for k=4 solutions
fig, ax = plt.subplots()

overlay_img = image.math_img(
    "img83 + img84 + img85", img83=nifti_img83, img84=nifti_img84, img85=nifti_img85
)

plotting.plot_roi(
    overlay_img, cut_coords=(50, -57, 4), cmap=cmap, alpha=0.5, colorbar=True, axes=ax
)

# Save the figure
output_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"
output_file = os.path.join(output_dir, "atlas8.png")
fig.savefig(output_file)

print(f"Figure saved as {output_file}")
"""
