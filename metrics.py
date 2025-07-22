import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    rand_score,
    silhouette_score,
)


def silhouette(data, labels, metric="euclidean"):
    silhouette_scores = []
    for i in range(len(labels)):
        silhouette_scores.append(silhouette_score(data, labels[i], metric=metric))

    return silhouette_scores


def hierarchy_index(labels):
    hierarchy_indexs = []

    for i in range(len(labels) - 1):
        tmp = np.zeros((int(np.max(labels[i])) + 1, int(np.max(labels[i + 1])) + 1))
        for j in range(int(np.max(labels[i]))):
            tmp_ind1 = np.where(labels[i] == j + 1)[0]
            for k in range(int(np.max(labels[i + 1]))):
                tmp_ind2 = np.where(labels[i + 1] == k + 1)[0]
                tmp[j, k] = len(np.intersect1d(tmp_ind1, tmp_ind2)) / len(labels[i])

        tmp_sum = np.sum(tmp, axis=0)
        tmp_sum[tmp_sum == 0] = 1e-6
        hierarchy_indexs.append(
            (100 * (1 - (1 / len(labels[i + 1])) * np.sum(np.max(tmp, axis=0) / tmp_sum))) - 99
        )
    return hierarchy_indexs


"""def cluster_consistency(labels):
    cluster_consistencys_mean = []
    cluster_consistencys_min = []

    for i in range(len(labels) - 1):
        tmp = np.zeros((int(np.max(labels[i])) + 1, int(np.max(labels[i + 1])) + 1))
        for j in range(int(np.max(labels[i]))):
            tmp_ind1 = np.where(labels[i] == j + 1)[0]
            for k in range(int(np.max(labels[i + 1]))):
                tmp_ind2 = np.where(labels[i + 1] == k + 1)[0]
                tmp[j, k] = len(np.intersect1d(tmp_ind1, tmp_ind2)) / len(labels[i])

        cluster_consistencys_mean.append(np.average(np.max(tmp, axis=1)))
        cluster_consistencys_min.append(np.average(np.max(tmp, axis=1)))

    return cluster_consistencys_mean, cluster_consistencys_min"""


def cluster_separation(data, labels):
    cluster_separations = []
    for i in range(len(labels)):
        cluster_separations.append(davies_bouldin_score(data, labels[i]))

    return cluster_separations


def variance_ratio(data, labels):
    variance_ratios = []
    for i in range(len(labels)):
        variance_ratios.append(calinski_harabasz_score(data, labels[i]))

    return variance_ratios
