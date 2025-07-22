import argparse
import os
import os.path as op
from glob import glob

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from nilearn.connectome import ConnectivityMeasure
from nimare.correct import FWECorrector
from nimare.io import convert_sleuth_to_dataset
from nimare.meta.cbma import ALE
from nimare.meta.kernel import ALEKernel
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans

from utils import get_peaks, plot_dendrogram, thresh_img

project_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"

"""
affiliation = glob(op.join(project_dir, "text-files_old", "*Affiliation*.txt"))
other = glob(op.join(project_dir, "text-files_old", "*Others*.txt"))
self = glob(op.join(project_dir, "text-files_old", "*Self*.txt"))
soccomm = glob(op.join(project_dir, "text-files_old", "*Soc_Comm*.txt"))

run_ales = False
if run_ales:
    for tmp_txts in [affiliation, other, self, soccomm]:
        prefix = "_".join(op.basename(tmp_txts[0]).split("_")[:-1])
        print(prefix)
        output_dir = op.join(project_dir, "derivatives", "ale", prefix)

        if not op.isfile(
            op.join(
                output_dir,
                "{prefix}_logp_level-cluster_corr-FWE_method-montecarlo.nii.gz".format(
                    prefix=prefix
                ),
            )
        ):
            os.makedirs(output_dir, exist_ok=True)

            dset = convert_sleuth_to_dataset(tmp_txts, target="mni152_2mm")

            ale = ALE()

            results = ale.fit(dset)
            corr = FWECorrector(method="montecarlo", n_iters=5000, voxel_thresh=0.001, n_cores=4)
            cres = corr.transform(results)

            cres.save_maps(output_dir=output_dir, prefix=prefix)

            dset.save(op.join(output_dir, prefix + ".pkl.gz"))

        z_img_logp = nib.load(
            op.join(
                output_dir,
                "{prefix}_logp_level-cluster_corr-FWE_method-montecarlo.nii.gz".format(
                    prefix=prefix
                ),
            )
        )
        z_img = nib.load(op.join(output_dir, "{prefix}_z.nii.gz".format(prefix=prefix)))
        z_img_thresh = thresh_img(z_img_logp, z_img, 0.001)
        nib.save(
            z_img_thresh,
            op.join(output_dir, "{prefix}_z_corr-cFWE_thresh-001.nii.gz".format(prefix=prefix)),
        )

        get_peaks(
            op.join(output_dir, "{prefix}_z_corr-cFWE_thresh-001.nii.gz".format(prefix=prefix)),
            output_dir,
        ) """

# clustering
all_txts = glob(op.join(project_dir, "text-files_old", "*.txt"))

output_dir = op.join(project_dir, "derivatives", "clustering_20220610_1")
os.makedirs(output_dir, exist_ok=True)

dset = convert_sleuth_to_dataset(all_txts, target="mni152_2mm")
dset.save(op.join(output_dir, "social-rdoc.pkl.gz"))

k = ALEKernel()
time_series = np.transpose(k.transform(dset, return_type="array"))
correlation = ConnectivityMeasure(kind="correlation")
corrmat = correlation.fit_transform([time_series])[0]

plot_dendrogram = True
if plot_dendrogram:
    clustering = AgglomerativeClustering(
        n_clusters=None,
        compute_full_tree=True,
        linkage="ward",
        distance_threshold=0,
        affinity="euclidean",
    )
    corrmat_clustering = clustering.fit(corrmat)
    fig = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    plt.title("RDoC Social - Hierarchical Clustering Dendrogram")
    plot_dendrogram(corrmat_clustering)  # , orientation="right"
    plt.savefig(op.join(output_dir, "hierarchical_input-corrmat.png"))
    plt.close()

"""
for d in range(2, 8, 1):

    clustering = KMeans(n_clusters=d, n_init=100, max_iter=1000, random_state=0)
    corrmat_cluster_labels = clustering.fit(1 - corrmat).labels_
    tmp_output_dir = op.join(output_dir, "cluster_{}".format(d))
    os.makedirs(tmp_output_dir, exist_ok=True)

    for a in range(np.max(corrmat_cluster_labels) + 1):

        tmp_dset = dset.slice(dset.annotations.id[np.where(corrmat_cluster_labels == a)[0]])

        ale = ALE()

        results = ale.fit(tmp_dset)
        corr = FWECorrector(method="montecarlo", n_iters=5000, voxel_thresh=0.001, n_cores=12)
        cres = corr.transform(results)

        cres.save_maps(output_dir=tmp_output_dir, prefix="cluster_{}".format(a))

        tmp_dset.save(op.join(tmp_output_dir, "cluster_{}.pkl.gz".format(a)))

        z_img_logp = nib.load(
            op.join(
                tmp_output_dir,
                "{prefix}_logp_desc-mass_level-cluster_corr-FWE_method-montecarlo.nii.gz".format(
                    prefix="cluster_{}".format(a)
                ),
            )
        )
        z_img = nib.load(
            op.join(tmp_output_dir, "{prefix}_z.nii.gz".format(prefix="cluster_{}".format(a)))
        )
        z_img_thresh = thresh_img(z_img_logp, z_img, 0.001)
        nib.save(
            z_img_thresh,
            op.join(
                tmp_output_dir,
                "{prefix}_z_corr-cFWE_thresh-001.nii.gz".format(prefix="cluster_{}".format(a)),
            ),
        )

        get_peaks(
            op.join(
                tmp_output_dir,
                "{prefix}_z_corr-cFWE_thresh-001.nii.gz".format(prefix="cluster_{}".format(a)),
            ),
            tmp_output_dir,
        )

        # decode(op.join(tmp_output_dir, '{prefix}_z.nii.gz'.format(prefix='cluster_{}'.format(a))))
"""
