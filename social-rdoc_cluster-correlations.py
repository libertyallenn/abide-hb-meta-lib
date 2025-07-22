import os
from glob import glob

import matplotlib.pyplot as plt

"""import nibabel as nib"""
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from nilearn.datasets import load_mni152_brain_mask
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map
from sklearn.metrics import pairwise_distances


# for the images on x-axiss
def offset_y_image(coord, imgs, ax):
    img = plt.imread(imgs[coord])
    zoom = 0.45
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax

    ab = AnnotationBbox(
        im,
        (coord, 3),
        xybox=(0, -zoom * (np.shape(img)[0]) - 15),
        frameon=False,
        xycoords="data",
        boxcoords="offset points",
        pad=0,
    )

    ax.add_artist(ab)


def offset_x_image(coord, imgs, ax):
    img = plt.imread(imgs[coord])
    zoom = 0.45
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax

    ab = AnnotationBbox(
        im,
        (0, coord),
        xybox=(-zoom * (np.shape(img)[1]) - 45, 0),
        frameon=False,
        xycoords="data",
        boxcoords="offset points",
        pad=0,
    )

    ax.add_artist(ab)


project_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"

ale_analysis = os.path.join(project_dir, "derivatives", "ale")
ale_analyses = ["Affiliation", "Others", "Self", "Soc_Comm"]

clustering_analysis = os.path.join(project_dir, "k_clustering", "clustering_856")
clustering_analyses = glob(os.path.join(clustering_analysis, "cluster_*"))

mask_img = load_mni152_brain_mask()
masker = NiftiMasker(mask_img=mask_img, memory="nilearn_cache", memory_level=1)
masker = masker.fit()

label_img_data = []
thresh_img_paths = []
for aa in ale_analyses:
    label_stat_image = os.path.join(
        ale_analysis, "{}_Co-coded".format(aa), "{}_Co-coded_stat.nii.gz".format(aa)
    )
    label_thresh_image = os.path.join(
        ale_analysis,
        "{}_Co-coded".format(aa),
        "{}_Co-coded_z_corr-cFWE_thresh-001.nii.gz".format(aa),
    )
    label_img_data.append(masker.transform(label_stat_image))
    plot_stat_map(
        label_thresh_image,
        cmap='Greens',
        display_mode="z",
        cut_coords=1,
        annotate=False,
        draw_cross=False,
        colorbar=False,
        output_file=label_thresh_image.replace(".nii.gz", ".png"),
    )
    thresh_img_paths.append(label_thresh_image.replace(".nii.gz", ".png"))

label_img_data = np.array(label_img_data).squeeze(axis=1)

for ca in clustering_analyses:
    dim = int(os.path.basename(ca).split("_")[1])
    cluster_img_data = []
    thresh_cluster_paths = []
    for d in range(dim):
        cluster_stat_image = os.path.join(ca, "cluster_{}_stat.nii.gz".format(d))
        cluster_thresh_image = os.path.join(
            ca, "cluster_{}_z_corr-cFWE_thresh-001.nii.gz".format(d)
        )
        cluster_img_data.append(masker.transform(cluster_stat_image))
        plot_stat_map(
            cluster_thresh_image,
            cmap='Blues',
            display_mode="z",
            cut_coords=1,
            annotate=False,
            draw_cross=False,
            colorbar=False,
            output_file=cluster_thresh_image.replace(".nii.gz", ".png"),
        )
        thresh_cluster_paths.append(cluster_thresh_image.replace(".nii.gz", ".png"))

    cluster_img_data = np.array(cluster_img_data).squeeze(axis=1)

    corr_mat = pairwise_distances(label_img_data, cluster_img_data, metric="correlation")

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.tight_layout(pad=10)

    im = ax.imshow(corr_mat, cmap="Blues_r")
    fig.colorbar(im)
    ax.set_xticks(np.arange(dim))
    ax.set_xticklabels(["Cluster {}".format(i) for i in range(dim)])
    ax.tick_params(axis="x", which="major", pad=26)
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(ale_analyses, rotation=90, fontsize=16, ha="center", rotation_mode="anchor")
    ax.tick_params(axis="y", which="major", pad=26)

    for i in range(len(ale_analyses)):
        offset_x_image(i, thresh_img_paths, ax)
    for i in range(dim):
        offset_y_image(i, thresh_cluster_paths, ax)

    plt.show()
    plt.savefig(
        os.path.join(clustering_analysis, ca, "label_img_correlations_cluster-{}.png".format(dim))
    )
    plt.close()
