import os.path as op
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nimare.io import convert_sleuth_to_dataset
from nimare.meta.kernel import ALEKernel
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# define variables and data
project_dir = "/home/data/nbc/misc-projects/meta-analyses/Hampson_rdoc-meta-analysis"
all_txts = glob(op.join(project_dir, "text-files", "*.txt"))
output_dir = op.join(project_dir)
target = "mni152_2mm"

dset = convert_sleuth_to_dataset(all_txts, target)
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
    plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    dendrogram(linkage(corrmat, method="ward"))
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Experiments")
    plt.ylabel("Cophenetic Distance")
    plt.savefig(op.join(output_dir, "dendogram_test1.png"))
    plt.close()
