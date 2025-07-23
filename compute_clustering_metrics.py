import os.path as op

import matplotlib.pyplot as plt
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nimare.dataset import Dataset
from nimare.meta.kernel import ALEKernel

import metrics

project_dir = "/home/data/nbc/misc-projects/meta-analyses/abide-hb-meta"
clustering_dir = op.join(project_dir, "derivatives", "k_clustering")

dset = Dataset.load(op.join(clustering_dir, "social-rdoc.pkl.gz"))
k = ALEKernel()
time_series = np.transpose(k.transform(dset, return_type="array"))
correlation = ConnectivityMeasure(kind="correlation")
corrmat = correlation.fit_transform([time_series])[0]

dset_ids = dset.ids

labels = []
for d in range(2, 9, 1):
    cluster_ids = np.zeros(len(dset_ids))
    for c in range(d):
        tmp_dset_ids = Dataset.load(
            op.join(clustering_dir, "cluster_{}".format(d), "cluster_{}.pkl.gz".format(c))
        ).ids
        tmp_cluster_idx = np.intersect1d(dset_ids, tmp_dset_ids, return_indices=True)[1]
        cluster_ids[tmp_cluster_idx] = c + 1
    labels.append(cluster_ids)

for d in range(len(labels)):
    print(d)
    print(labels[d])
    print(np.max(labels[d]))
    print(np.min(labels[d]))
"""sils = metrics.silhouette(1 - corrmat, labels, metric="euclidean")
plt.plot(range(2, 9, 1), sils, color="crimson")  # Dark blue line
plt.scatter(range(2, 9, 1), sils, c="crimson", marker=".", s=100)  # Bright pink star
plt.xlabel("Cluster", fontweight="bold", fontsize=12)
plt.ylabel("Average Silhouette Coefficient", fontweight="bold", fontsize=12)
plt.title("Silhouette Score", fontweight="bold", fontsize=12)
plt.grid(True, color="b", linestyle="dashdot", linewidth=0.25)  # Grey grid lines
plt.xticks(fontsize=12)
plt.yticks(fontsize=9)
plt.savefig("silhouette_scores.png", dpi=300, fig_size=(8, 6))
plt.close()

hi = metrics.hierarchy_index(labels)
plt.plot(range(3, 9, 1), hi, color="purple")  # Dark blue line
plt.scatter(range(3, 9, 1), hi, c="purple", marker=".", s=100)  # Bright pink star
plt.xlabel("Cluster", fontweight="bold", fontsize=12)
plt.ylabel("Experiments lost", fontweight="bold", fontsize=12)
plt.title("Hierarchy Index", fontweight="bold", fontsize=12)
plt.grid(True, color="b", linestyle="dashdot", linewidth=0.25)  # Grey grid lines
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("hierarchy_index.png", dpi=300, fig_size=(8, 6))
plt.close()

vr = metrics.variance_ratio(1 - corrmat, labels)
plt.plot(range(2, 9, 1), vr, color="forestgreen")  # Dark blue line
plt.scatter(range(2, 9, 1), vr, c="forestgreen", marker=".", s=100)  # Bright pink star
plt.xlabel("Cluster", fontweight="bold", fontsize=12)
plt.ylabel("Variation of information", fontweight="bold", fontsize=12)
plt.title("Variance Ratio", fontweight="bold", fontsize=12)
plt.grid(True, color="b", linestyle="dashdot", linewidth=0.25)  # Grey grid lines
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("variance_ratio.png", dpi=300, fig_size=(8, 6))"""

cs = metrics.cluster_separation(1 - corrmat, labels)
plt.plot(range(2, 9, 1), cs, color="royalblue")  # Dark blue line
plt.scatter(range(2, 9, 1), cs, c="royalblue", marker=".", s=100)  # Bright pink star
plt.xlabel("Cluster", fontweight="bold", fontsize=12)
plt.ylabel("Davies-Bouldin score", fontweight="bold", fontsize=12)
plt.title("Cluster Separation", fontweight="bold", fontsize=12)
plt.grid(True, color="b", linestyle="dashdot", linewidth=0.25)  # Grey grid lines
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("cluster_separation.png", dpi=300, fig_size=(8, 6))
plt.close()


"""cc = metrics.cluster_consistency(labels)
plt.plot(range(3,9,1), cc)
plt.xlabel('Cluster')
plt.ylabel('Cluster Consistency')
plt.title('Cluster Consistency)
plt.savefig('cluster_consistency.png')
plt.close()"""

"""a = [196, 139, 82, 63, 32, 35, 43]
b = [426, 284, 213, 170.4, 142, 121.7, 106.5]
y = [a, b]
x = list(range(2, 9))
width = 0.35  # width of the bars
offset = 0.37  # offset between the bars

fig, ax = plt.subplots()

ax.bar(
    [i - offset / 2 for i in x],
    height=b,
    width=width,
    color="royalblue",
    label="Experiment Averages",
)

ax.bar(
    [i + offset / 2 for i in x],
    height=a,
    width=width,
    color="crimson",
    label="Experiment Minimum",
)

ax.set_xlabel("# of clusters", fontweight="bold")
ax.set_ylabel("# of experiments per cluster", fontweight="bold")
ax.set_title("Cluster Consistency Across Cluster Solutions", fontweight="bold")
ax.legend()  # Add legend

# Manually plot the grid lines
ax.grid(axis="y", color="gray", linestyle="--", linewidth=0.5)

# Add value labels above each bar
for i, j in enumerate(x):
    ax.text(j - offset / 2, b[i] + 5, str(b[i]), ha="center")
    ax.text(j + offset / 2, a[i] + 5, str(a[i]), ha="center")

plt.savefig("cluster_consistency.png", dpi=300)
plt.close()"""

"""cs = metrics.cluster_separation(1-corrmat, labels)
plt.plot(range(2,9,1), cs)
plt.xlabel('Cluster')
plt.ylabel('Cluster Separation')
plt.title('Cluster Separation Across Cluster Solutions')
plt.savefig('cluster_separation.png')
plt.close()"""
