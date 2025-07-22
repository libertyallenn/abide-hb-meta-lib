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
from sklearn.cluster import AgglomerativeClustering, KMeans

from utils import get_peaks, thresh_img

project_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"

# clustering
all_txts = glob(op.join(project_dir, "text-files", "*.txt"))

output_dir = op.join(project_dir, "k_clustering", "clustering_856")
os.makedirs(output_dir, exist_ok=True)

dset = convert_sleuth_to_dataset(all_txts, target="mni152_2mm")
dset.save(op.join(output_dir, "social-rdoc.pkl.gz"))

k = ALEKernel()
time_series = np.transpose(k.transform(dset, return_type="array"))
correlation = ConnectivityMeasure(kind="correlation")
corrmat = correlation.fit_transform([time_series])[0]

for d in range(1, 2, 1):

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
