import argparse
import fastcluster

import os
import os.path as op
from glob import glob

import pandas as pd
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from nilearn.maskers import NiftiMasker
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from utils import get_peaks, thresh_img


def get_parser():
    p = argparse.ArgumentParser(description="Habenula RSFC clustering workflow")
    p.add_argument(
        "--project_dir",
        required=True,
        help="Path to project directory (used for metadata and relative references)",
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Base directory containing RSFC connectivity maps (e.g., /home/data/.../rsfc-habenula)",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Directory where outputs (figures, cluster results) will be written",
    )
    p.add_argument(
        "--method",
        choices=[
            "kmeans_no_pca",
            "kmeans_with_pca",
            "spectral",
            "kmeans_similarity",
            "all",
        ],
        default="all",
        help="Clustering method to run: kmeans_no_pca, kmeans_with_pca, "
        "spectral, kmeans_similarity, or all",
    )
    return p


class HabenulaClustering:
    def kmeans_no_pca(self, k, n_init=100, max_iter=1000, random_state=42):
        """KMeans clustering directly on the data matrix (no PCA)."""
        print(f"Running KMeans without PCA (k={k})...")
        kmeans = KMeans(
            n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state
        )
        labels = kmeans.fit_predict(self.data_matrix)
        silhouette = silhouette_score(self.data_matrix, labels)
        print(f"Silhouette score (no PCA): {silhouette:.3f}")
        return labels, kmeans, silhouette

    def kmeans_with_pca(
        self, k, n_components=50, n_init=100, max_iter=1000, random_state=42
    ):
        """KMeans clustering after PCA dimensionality reduction."""
        print(f"Running KMeans with PCA (k={k}, n_components={n_components})...")
        pca = PCA(
            n_components=min(
                n_components, self.data_matrix.shape[0] // 2, self.data_matrix.shape[1]
            )
        )
        reduced_data = pca.fit_transform(self.data_matrix)
        kmeans = KMeans(
            n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state
        )
        labels = kmeans.fit_predict(reduced_data)
        silhouette = silhouette_score(reduced_data, labels)
        print(f"Silhouette score (with PCA): {silhouette:.3f}")
        return labels, kmeans, silhouette, pca

    def spectral_clustering(self, k, affinity="nearest_neighbors", random_state=42):
        """Spectral clustering on the data matrix."""
        print(f"Running Spectral Clustering (k={k}, affinity={affinity})...")
        spectral = SpectralClustering(
            n_clusters=k,
            affinity=affinity,
            random_state=random_state,
            assign_labels="kmeans",
        )
        labels = spectral.fit_predict(self.data_matrix)
        # Silhouette score may not be meaningful for all affinity types
        try:
            silhouette = silhouette_score(self.data_matrix, labels)
            print(f"Silhouette score (spectral): {silhouette:.3f}")
        except Exception as e:
            print(f"Could not compute silhouette score for spectral clustering: {e}")
            silhouette = None
        return labels, spectral, silhouette

    def kmeans_similarity_matrix(self, k, n_init=100, max_iter=1000, random_state=42):
        """KMeans clustering directly on the similarity matrix."""
        print(f"Running KMeans on similarity matrix (k={k})...")
        kmeans = KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        labels = kmeans.fit_predict(self.similarity_matrix)
        silhouette = silhouette_score(self.similarity_matrix, labels)
        print(f"Silhouette score (similarity matrix): {silhouette:.3f}")
        return labels, kmeans, silhouette

    def perform_all_clustering_methods(self, k_values, n_components_pca=50):
        """Perform all four clustering methods and save results separately."""

        self.all_clustering_results = {
            "kmeans_no_pca": {},
            "kmeans_with_pca": {},
            "spectral": {},
            "kmeans_similarity": {},
        }

        for k in k_values:
            print(f"\n=== Clustering with k={k} ===")

            # 1. KMeans without PCA
            labels_no_pca, model_no_pca, sil_no_pca = self.kmeans_no_pca(k)
            self.all_clustering_results["kmeans_no_pca"][k] = {
                "labels": labels_no_pca,
                "model": model_no_pca,
                "silhouette": sil_no_pca,
            }
            self._save_clustering_results(
                k, labels_no_pca, "kmeans_no_pca", self.kmeans_no_pca_dir
            )

            # 2. KMeans with PCA
            labels_pca, model_pca, sil_pca, pca_model = self.kmeans_with_pca(
                k, n_components=n_components_pca
            )
            self.all_clustering_results["kmeans_with_pca"][k] = {
                "labels": labels_pca,
                "model": model_pca,
                "silhouette": sil_pca,
                "pca_model": pca_model,
            }
            self._save_clustering_results(
                k, labels_pca, "kmeans_with_pca", self.kmeans_pca_dir
            )

            # 3. Spectral clustering
            labels_spectral, model_spectral, sil_spectral = self.spectral_clustering(k)
            self.all_clustering_results["spectral"][k] = {
                "labels": labels_spectral,
                "model": model_spectral,
                "silhouette": sil_spectral,
            }
            self._save_clustering_results(
                k, labels_spectral, "spectral", self.spectral_dir
            )

            # 4. KMeans on similarity matrix
            labels_sim, model_sim, sil_sim = self.kmeans_similarity_matrix(k)
            self.all_clustering_results["kmeans_similarity"][k] = {
                "labels": labels_sim,
                "model": model_sim,
                "silhouette": sil_sim,
            }
            self._save_clustering_results(
                k, labels_sim, "kmeans_similarity", self.kmeans_similarity_dir
            )

        return self

    def perform_single_clustering_method(
        self, method_name, k_values, n_components_pca=50
    ):
        """Perform a single clustering method and save results."""

        print(f"Running {method_name} clustering...")

        self.clustering_results = {}

        for k in k_values:
            print(f"\n=== {method_name} Clustering with k={k} ===")

            if method_name == "kmeans_no_pca":
                labels, model, silhouette = self.kmeans_no_pca(k)
                output_dir = self.kmeans_no_pca_dir
            elif method_name == "kmeans_with_pca":
                labels, model, silhouette, pca_model = self.kmeans_with_pca(
                    k, n_components=n_components_pca
                )
                output_dir = self.kmeans_pca_dir
            elif method_name == "spectral":
                labels, model, silhouette = self.spectral_clustering(k)
                output_dir = self.spectral_dir
            elif method_name == "kmeans_similarity":
                labels, model, silhouette = self.kmeans_similarity_matrix(k)
                output_dir = self.kmeans_similarity_dir
            else:
                raise ValueError(f"Unknown method: {method_name}")

            self.clustering_results[k] = {
                "labels": labels,
                "model": model,
                "silhouette": silhouette,
            }

            # Save results
            self._save_clustering_results(k, labels, method_name, output_dir)

        # Create visualizations for this method
        self._visualize_single_method(method_name, k_values)

        return self

    def _visualize_single_method(self, method_name, k_values):
        """Create visualizations for a single clustering method."""

        print(f"\nCreating visualizations for {method_name}...")

        # Set the appropriate figures directory
        if method_name == "kmeans_no_pca":
            figures_dir = self.kmeans_no_pca_figures
        elif method_name == "kmeans_with_pca":
            figures_dir = self.kmeans_pca_figures
        elif method_name == "spectral":
            figures_dir = self.spectral_figures
        elif method_name == "kmeans_similarity":
            figures_dir = self.kmeans_similarity_figures
        else:
            raise ValueError(f"Unknown method: {method_name}")

        for k in k_values:
            if k not in self.clustering_results:
                continue

            labels = self.clustering_results[k]["labels"]

            # Create cluster visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. PCA visualization (for visualization purposes only)
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(self.data_matrix)

            axes[0, 0].scatter(
                data_2d[:, 0], data_2d[:, 1], c=labels, cmap="tab10", alpha=0.7
            )
            axes[0, 0].set_xlabel(
                f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)"
            )
            axes[0, 0].set_ylabel(
                f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
            )
            axes[0, 0].set_title(f"{method_name} Clusters in PCA Space (k={k})")

            # 2. Cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            axes[0, 1].bar(unique, counts, alpha=0.7)
            axes[0, 1].set_xlabel("Cluster ID")
            axes[0, 1].set_ylabel("Number of Participants")
            axes[0, 1].set_title(f"{method_name} Cluster Sizes (k={k})")

            # 3. Similarity matrix with cluster ordering
            cluster_order = np.argsort(labels)
            ordered_similarity = self.similarity_matrix[
                np.ix_(cluster_order, cluster_order)
            ]
            ordered_labels = labels[cluster_order]

            im = axes[1, 0].imshow(
                ordered_similarity, cmap="RdBu_r", vmin=-0.4, vmax=0.4
            )
            axes[1, 0].set_title(
                f"{method_name} Similarity Matrix (Ordered by Clusters, k={k})"
            )
            axes[1, 0].set_xlabel("Participant (Ordered)")
            axes[1, 0].set_ylabel("Participant (Ordered)")
            plt.colorbar(im, ax=axes[1, 0])

            # Add cluster boundaries
            cluster_boundaries = []
            current_cluster = ordered_labels[0]
            for i, cluster in enumerate(ordered_labels[1:], 1):
                if cluster != current_cluster:
                    cluster_boundaries.append(i - 0.5)
                    current_cluster = cluster

            for boundary in cluster_boundaries:
                axes[1, 0].axhline(boundary, color="white", linewidth=2)
                axes[1, 0].axvline(boundary, color="white", linewidth=2)

            # 4. Within-cluster similarity distribution
            within_cluster_sims = []
            between_cluster_sims = []

            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if labels[i] == labels[j]:
                        within_cluster_sims.append(self.similarity_matrix[i, j])
                    else:
                        between_cluster_sims.append(self.similarity_matrix[i, j])

            axes[1, 1].hist(
                within_cluster_sims,
                bins=30,
                alpha=0.7,
                label="Within cluster",
                density=True,
            )
            axes[1, 1].hist(
                between_cluster_sims,
                bins=30,
                alpha=0.7,
                label="Between cluster",
                density=True,
            )
            axes[1, 1].set_xlabel("Similarity")
            axes[1, 1].set_ylabel("Density")
            axes[1, 1].set_title(f"{method_name} Similarity Distributions (k={k})")
            axes[1, 1].legend()

            plt.tight_layout()
            plt.savefig(
                op.join(figures_dir, f"clusters_k_{k}_{method_name}_visualization.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        return self

    def _save_clustering_results(self, k, labels, method_name, output_dir):
        """Save clustering results for a specific method."""

        # Create output directory for this k and method
        k_output_dir = op.join(output_dir, f"k_{k}_clusters")
        os.makedirs(k_output_dir, exist_ok=True)

        # Analyze each cluster
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_subjects = [
                self.subject_ids[i] for i in range(len(labels)) if cluster_mask[i]
            ]

            print(
                f"    {method_name} Cluster {cluster_id}: {len(cluster_subjects)} participants"
            )

            # Compute cluster centroid (mean beta values)
            cluster_data = self.data_matrix[cluster_mask]
            cluster_centroid = np.mean(cluster_data, axis=0)
            centroid_img = self.masker.inverse_transform(cluster_centroid)

            # Save centroid beta map
            centroid_path = op.join(
                k_output_dir, f"cluster_{cluster_id}_centroid_beta.nii.gz"
            )
            nib.save(centroid_img, centroid_path)

            # Threshold beta map (absolute threshold)
            beta_thresh_value = 0.2
            beta_thresh_img = thresh_img(centroid_img, beta_thresh_value)
            beta_thresh_path = op.join(
                k_output_dir,
                f"cluster_{cluster_id}_beta_thresh-{beta_thresh_value}.nii.gz",
            )
            nib.save(beta_thresh_img, beta_thresh_path)

            # Convert beta centroid to Z-map
            centroid_data = centroid_img.get_fdata()
            z_data = (centroid_data - self.global_mean) / self.global_std
            z_img = nib.Nifti1Image(z_data, centroid_img.affine)
            z_path = op.join(k_output_dir, f"cluster_{cluster_id}_centroid_zmap.nii.gz")
            nib.save(z_img, z_path)

            # Threshold Z-map (statistical threshold)
            z_thresh_value = 3.09  # ~p<0.001
            z_thresh_img = thresh_img(z_img, z_thresh_value)
            z_thresh_path = op.join(
                k_output_dir, f"cluster_{cluster_id}_z_thresh-{z_thresh_value}.nii.gz"
            )
            nib.save(z_thresh_img, z_thresh_path)

            # Save participant list
            subjects_file = op.join(
                k_output_dir, f"cluster_{cluster_id}_participants.txt"
            )
            with open(subjects_file, "w") as f:
                for subj in cluster_subjects:
                    f.write(f"{subj}\n")

        # Save cluster assignments
        assignments_df = pd.DataFrame(
            {"subject_id": self.subject_ids, "cluster": labels}
        )
        assignments_file = op.join(
            k_output_dir, f"cluster_assignments_k_{k}_{method_name}.csv"
        )
        assignments_df.to_csv(assignments_file, index=False)

    def visualize_all_clustering_methods(self, k_values):
        """Create visualizations for all three clustering methods."""

        for method_name in ["kmeans_no_pca", "kmeans_with_pca", "spectral"]:
            print(f"\nCreating visualizations for {method_name}...")

            # Set the appropriate figures directory
            if method_name == "kmeans_no_pca":
                figures_dir = self.kmeans_no_pca_figures
            elif method_name == "kmeans_with_pca":
                figures_dir = self.kmeans_pca_figures
            else:
                figures_dir = self.spectral_figures

            for k in k_values:
                if k not in self.all_clustering_results[method_name]:
                    continue

                labels = self.all_clustering_results[method_name][k]["labels"]

                # Create cluster visualization
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))

                # 1. PCA visualization (for visualization purposes only)
                pca = PCA(n_components=2)
                data_2d = pca.fit_transform(self.data_matrix)

                axes[0, 0].scatter(
                    data_2d[:, 0], data_2d[:, 1], c=labels, cmap="tab10", alpha=0.7
                )
                axes[0, 0].set_xlabel(
                    f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)"
                )
                axes[0, 0].set_ylabel(
                    f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
                )
                axes[0, 0].set_title(f"{method_name} Clusters in PCA Space (k={k})")

                # 2. Cluster sizes
                unique, counts = np.unique(labels, return_counts=True)
                axes[0, 1].bar(unique, counts, alpha=0.7)
                axes[0, 1].set_xlabel("Cluster ID")
                axes[0, 1].set_ylabel("Number of Participants")
                axes[0, 1].set_title(f"{method_name} Cluster Sizes (k={k})")

                # 3. Similarity matrix with cluster ordering
                cluster_order = np.argsort(labels)
                ordered_labels = labels[cluster_order]

                im = axes[1, 0].imshow(
                    self.similarity_matrix, cmap="RdBu_r", vmin=-0.4, vmax=0.4
                )
                axes[1, 0].set_title(
                    f"{method_name} Similarity Matrix (Ordered by Clusters, k={k})"
                )
                axes[1, 0].set_xlabel("Participant (Ordered)")
                axes[1, 0].set_ylabel("Participant (Ordered)")
                plt.colorbar(im, ax=axes[1, 0])

                # Add cluster boundaries
                cluster_boundaries = []
                current_cluster = ordered_labels[0]
                for i, cluster in enumerate(ordered_labels[1:], 1):
                    if cluster != current_cluster:
                        cluster_boundaries.append(i - 0.5)
                        current_cluster = cluster

                for boundary in cluster_boundaries:
                    axes[1, 0].axhline(boundary, color="white", linewidth=2)
                    axes[1, 0].axvline(boundary, color="white", linewidth=2)

                # 4. Within-cluster similarity distribution
                within_cluster_sims = []
                between_cluster_sims = []

                for i in range(len(labels)):
                    for j in range(i + 1, len(labels)):
                        if labels[i] == labels[j]:
                            within_cluster_sims.append(self.similarity_matrix[i, j])
                        else:
                            between_cluster_sims.append(self.similarity_matrix[i, j])

                axes[1, 1].hist(
                    within_cluster_sims,
                    bins=30,
                    alpha=0.7,
                    label="Within cluster",
                    density=True,
                )
                axes[1, 1].hist(
                    between_cluster_sims,
                    bins=30,
                    alpha=0.7,
                    label="Between cluster",
                    density=True,
                )
                axes[1, 1].set_xlabel("Similarity")
                axes[1, 1].set_ylabel("Density")
                axes[1, 1].set_title(f"{method_name} Similarity Distributions (k={k})")
                axes[1, 1].legend()

                plt.tight_layout()
                plt.savefig(
                    op.join(
                        figures_dir, f"clusters_k_{k}_{method_name}_visualization.png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()  # Close to save memory

        return self

    def compare_clustering_methods(self, k_values):
        """Create comparison visualizations across all four clustering methods."""

        print("Creating method comparison visualizations...")

        # Prepare data for comparison
        methods = ["kmeans_no_pca", "kmeans_with_pca", "spectral", "kmeans_similarity"]
        method_labels = [
            "KMeans (no PCA)",
            "KMeans (with PCA)",
            "Spectral",
            "KMeans (similarity)",
        ]

        for k in k_values:
            # Check if all methods have results for this k
            if not all(k in self.all_clustering_results[method] for method in methods):
                continue

            fig, axes = plt.subplots(2, 4, figsize=(24, 12))

            for i, (method, method_label) in enumerate(zip(methods, method_labels)):
                labels = self.all_clustering_results[method][k]["labels"]
                silhouette = self.all_clustering_results[method][k]["silhouette"]

                # PCA visualization for each method (top row)
                pca = PCA(n_components=2)
                data_2d = pca.fit_transform(self.data_matrix)

                axes[0, i].scatter(
                    data_2d[:, 0], data_2d[:, 1], c=labels, cmap="tab10", alpha=0.7
                )
                axes[0, i].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
                axes[0, i].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
                if silhouette is not None:
                    axes[0, i].set_title(
                        f"{method_label} (k={k})\nSilhouette: {silhouette:.3f}"
                    )
                else:
                    axes[0, i].set_title(f"{method_label} (k={k})")

                # Cluster sizes (bottom row)
                unique, counts = np.unique(labels, return_counts=True)
                axes[1, i].bar(unique, counts, alpha=0.7)
                axes[1, i].set_xlabel("Cluster ID")
                axes[1, i].set_ylabel("Number of Participants")
                axes[1, i].set_title(f"{method_label} Cluster Sizes")

            plt.tight_layout()
            plt.savefig(
                op.join(self.output_dir, f"method_comparison_k_{k}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # Create silhouette score comparison across k values
        fig, ax = plt.subplots(figsize=(10, 6))

        for method, method_label in zip(methods, method_labels):
            k_vals = []
            sil_scores = []

            for k in sorted(k_values):
                if k in self.all_clustering_results[method]:
                    sil_score = self.all_clustering_results[method][k]["silhouette"]
                    if sil_score is not None:
                        k_vals.append(k)
                        sil_scores.append(sil_score)

            if k_vals:
                ax.plot(k_vals, sil_scores, marker="o", label=method_label)

        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score Comparison Across Methods")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            op.join(self.output_dir, "silhouette_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        return self

    """
    Class for clustering participants based on whole-brain habenula connectivity maps
    """

    def __init__(self, project_dir, base_rsfc_dir):
        self.project_dir = project_dir
        self.base_rsfc_dir = base_rsfc_dir

        # Initialize output directory paths (will be set properly in main())
        self.output_dir = None
        self.kmeans_no_pca_dir = None
        self.kmeans_pca_dir = None
        self.spectral_dir = None
        self.kmeans_similarity_dir = None
        self.kmeans_no_pca_figures = None
        self.kmeans_pca_figures = None
        self.spectral_figures = None
        self.kmeans_similarity_figures = None

        # Initialize data containers
        self.df = None
        self.data_matrix = None
        self.subject_ids = None
        self.map_paths = None
        self.masker = None

    def setup_output_directories(self, output_dir):
        """Set up output directory structure."""
        self.output_dir = output_dir

        # Create method-specific output directories
        self.kmeans_no_pca_dir = op.join(output_dir, "kmeans_no_pca")
        self.kmeans_pca_dir = op.join(output_dir, "kmeans_with_pca")
        self.spectral_dir = op.join(output_dir, "spectral_clustering")
        self.kmeans_similarity_dir = op.join(output_dir, "kmeans_similarity")

        # Create figures directories for each method
        self.kmeans_no_pca_figures = op.join(self.kmeans_no_pca_dir, "figures")
        self.kmeans_pca_figures = op.join(self.kmeans_pca_dir, "figures")
        self.spectral_figures = op.join(self.spectral_dir, "figures")
        self.kmeans_similarity_figures = op.join(self.kmeans_similarity_dir, "figures")

        # Create all output directories
        os.makedirs(self.kmeans_no_pca_dir, exist_ok=True)
        os.makedirs(self.kmeans_pca_dir, exist_ok=True)
        os.makedirs(self.spectral_dir, exist_ok=True)
        os.makedirs(self.kmeans_similarity_dir, exist_ok=True)
        os.makedirs(self.kmeans_no_pca_figures, exist_ok=True)
        os.makedirs(self.kmeans_pca_figures, exist_ok=True)
        os.makedirs(self.spectral_figures, exist_ok=True)
        os.makedirs(self.kmeans_similarity_figures, exist_ok=True)

        return self

    def load_data(self, metadata_file, group_filter="asd"):
        """Load and prepare connectivity maps for clustering"""

        print(f"Loading data from {metadata_file}")

        # Load metadata
        self.df = pd.read_csv(metadata_file, sep="\t", comment="#")

        # Filter by group if specified
        if group_filter:
            self.df = self.df[self.df["group"] == group_filter].copy()
            print(f"Filtered to {len(self.df)} {group_filter} participants")

        # Create full paths to connectivity maps
        self.df["map_path"] = self.df["InputFile"].str.replace(
            "^/rsfc/", f"{self.base_rsfc_dir}/", regex=True
        )

        # Keep only existing files
        existing_mask = self.df["map_path"].apply(op.exists)
        self.df = self.df[existing_mask]

        print(f"Found {len(self.df)} participants with valid connectivity maps")

        # Extract paths and subject IDs
        self.map_paths = self.df["map_path"].tolist()
        self.subject_ids = self.df["Subj"].tolist()

        # Verify all files can be loaded
        print("Verifying file integrity...")
        valid_files = []
        valid_subjects = []
        valid_paths = []

        for i, path in enumerate(self.map_paths):
            try:
                nib.load(path)
                valid_files.append(i)
                valid_subjects.append(self.subject_ids[i])
                valid_paths.append(path)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")

        self.map_paths = valid_paths
        self.subject_ids = valid_subjects
        print(f"Final dataset: {len(self.map_paths)} valid connectivity maps")

        return self

    def prepare_data_matrix(
        self, mask_strategy="whole-brain-template", standardize=True
    ):
        """Convert connectivity maps to data matrix for clustering"""

        print("Creating data matrix from connectivity maps...")

        # Initialize masker
        self.masker = NiftiMasker(
            mask_strategy=mask_strategy,
            standardize=standardize,
            memory_level=1,
            verbose=1,
        )

        # Transform maps to data matrix
        # Shape: (n_participants, n_voxels)
        self.data_matrix = self.masker.fit_transform(self.map_paths)

        # Compute global mean and std for later z-map calculations
        self.global_mean = np.mean(self.data_matrix)
        self.global_std = np.std(self.data_matrix)

        print(f"Data matrix shape: {self.data_matrix.shape}")
        print(f"Memory usage: ~{self.data_matrix.nbytes / 1e9:.2f} GB")

        return self

    def compute_similarity_matrix(self):
        """Compute participant similarity matrix"""

        print("Computing participant similarity matrix...")

        # Compute correlation matrix between participants
        self.similarity_matrix = np.corrcoef(self.data_matrix)

        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")

        return self

    def visualize_similarity(self):
        """Create visualizations of participant similarity"""

        print("Creating similarity visualizations...")

        # Create comprehensive similarity plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Basic heatmap
        im1 = axes[0, 0].imshow(
            self.similarity_matrix, cmap="RdBu_r", vmin=-0.4, vmax=0.4
        )
        axes[0, 0].set_title("Participant Similarity Matrix")
        axes[0, 0].set_xlabel("Participant Index")
        axes[0, 0].set_ylabel("Participant Index")
        plt.colorbar(im1, ax=axes[0, 0], label="Correlation")

        # 2. Seaborn heatmap
        sns.heatmap(
            self.similarity_matrix,
            cmap="RdBu_r",
            center=0,
            ax=axes[0, 1],
            square=True,
            cbar_kws={"label": "Correlation"},
            vmin=-0.4,
            vmax=0.4,
        )
        axes[0, 1].set_title("Similarity Heatmap")

        # 3. Hierarchical clustering dendrogram
        # Compute linkage
        condensed_dist = pdist(self.data_matrix, metric="correlation")
        linkage_matrix = linkage(condensed_dist, method="ward")

        dendrogram(linkage_matrix, ax=axes[1, 0], truncate_mode="level", p=10)
        axes[1, 0].set_title("Hierarchical Clustering Dendrogram")
        axes[1, 0].set_xlabel("Sample Index or (Cluster Size)")
        axes[1, 0].set_ylabel("Distance")

        # 4. Similarity distribution
        mask = np.triu(np.ones_like(self.similarity_matrix, dtype=bool), k=1)
        upper_tri_sims = self.similarity_matrix[mask]

        axes[1, 1].hist(upper_tri_sims, bins=50, alpha=0.7, edgecolor="black")
        axes[1, 1].axvline(
            np.mean(upper_tri_sims),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(upper_tri_sims):.3f}",
        )
        axes[1, 1].axvline(
            np.median(upper_tri_sims),
            color="orange",
            linestyle="--",
            label=f"Median: {np.median(upper_tri_sims):.3f}",
        )
        axes[1, 1].set_xlabel("Pairwise Similarity")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Distribution of Participant Similarities")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(
            op.join(self.output_dir, "similarity_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # Create separate hierarchically clustered heatmap
        plt.figure(figsize=(12, 10))
        g = sns.clustermap(
            self.similarity_matrix,
            cmap="RdBu_r",
            center=0,
            vmin=-0.4,
            vmax=0.4,
            figsize=(10, 8),
            cbar_kws={"label": "Correlation"},
        )
        g.ax_heatmap.set_title("Hierarchically Clustered Similarity Matrix")
        g.savefig(
            op.join(self.output_dir, "similarity_clustered.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        return self

    def determine_optimal_clusters(self, k_range, methods=["silhouette", "elbow"]):
        """Determine optimal number of clusters using similarity matrix (distance-based)"""

        print("Determining optimal number of clusters...")

        # Convert similarity to distance
        distance_matrix = 1 - self.similarity_matrix

        # Reduce dimensionality (PCA on distance matrix)
        pca = PCA(n_components=min(50, distance_matrix.shape[0] // 2))
        reduced_data = pca.fit_transform(distance_matrix)

        self.cluster_metrics = {}

        for k in k_range:
            print(f"  Testing k={k}...")

            kmeans = KMeans(n_clusters=k, n_init=100, max_iter=1000, random_state=42)
            labels = kmeans.fit_predict(reduced_data)

            # Silhouette
            if "silhouette" in methods:
                sil_score = silhouette_score(reduced_data, labels)
                if k not in self.cluster_metrics:
                    self.cluster_metrics[k] = {}
                self.cluster_metrics[k]["silhouette"] = sil_score

            # Elbow (inertia)
            if "elbow" in methods:
                inertia = kmeans.inertia_
                if k not in self.cluster_metrics:
                    self.cluster_metrics[k] = {}
                self.cluster_metrics[k]["elbow"] = inertia

        # Plot metrics
        fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5))
        if len(methods) == 1:
            axes = [axes]

        for i, method in enumerate(methods):
            k_values = list(self.cluster_metrics.keys())
            metric_values = [self.cluster_metrics[k][method] for k in k_values]

            axes[i].plot(k_values, metric_values, "bo-")
            axes[i].set_xlabel("Number of Clusters (k)")
            axes[i].set_ylabel(method.capitalize())
            axes[i].set_title(f"Cluster Validation: {method.capitalize()}")
            axes[i].grid(True, alpha=0.3)

            # Highlight optimal k
            if method == "silhouette":
                optimal_k = k_values[np.argmax(metric_values)]
                axes[i].axvline(
                    optimal_k,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Optimal k={optimal_k}",
                )
                axes[i].legend()

        plt.tight_layout()
        plt.savefig(
            op.join(self.output_dir, "cluster_validation.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # Print recommendations and save to file
        results_path = op.join(self.output_dir, "cluster_validation_scores.txt")
        with open(results_path, "w") as f:
            f.write("Cluster validation results:\n")
            for k in sorted(self.cluster_metrics.keys()):
                metrics_str = ", ".join(
                    [
                        f"{metric}: {value:.3f}"
                        for metric, value in self.cluster_metrics[k].items()
                    ]
                )
                print(f"  k={k}: {metrics_str}")
                f.write(f"k={k}: {metrics_str}\n")

            if "silhouette" in methods:
                k_values = list(self.cluster_metrics.keys())
                sil_scores = [self.cluster_metrics[k]["silhouette"] for k in k_values]
                optimal_k = k_values[np.argmax(sil_scores)]
                print(f"\nRecommended k based on silhouette score: {optimal_k}")
                f.write(f"\nRecommended k based on silhouette score: {optimal_k}\n")

        print(f"\nValidation scores saved to {results_path}")

        return self

    def perform_clustering(self, k_values):
        """Cluster participants using similarity matrix; still compute voxel centroids."""

        print("Performing k-means clustering (similarity-based)...")

        # Convert similarity to distance
        distance_matrix = 1 - self.similarity_matrix

        # PCA reduction
        pca = PCA(n_components=min(50, distance_matrix.shape[0] // 2))
        reduced_data = pca.fit_transform(distance_matrix)

        self.clustering_results = {}

        for k in k_values:
            print(f"  Clustering with k={k}...")

            # Cluster using reduced distance
            kmeans = KMeans(n_clusters=k, n_init=100, max_iter=1000, random_state=42)
            labels = kmeans.fit_predict(reduced_data)

            self.clustering_results[k] = {
                "labels": labels,
                "model": kmeans,
                "silhouette": silhouette_score(reduced_data, labels),
            }

            # Create output directory for this k
            k_output_dir = op.join(self.output_dir, f"k_{k}_clusters")
            os.makedirs(k_output_dir, exist_ok=True)

            # Analyze each cluster
            for cluster_id in range(k):
                cluster_mask = labels == cluster_id
                cluster_subjects = [
                    self.subject_ids[i] for i in range(len(labels)) if cluster_mask[i]
                ]

                print(f"    Cluster {cluster_id}: {len(cluster_subjects)} participants")

                # Compute cluster centroid (mean beta values)
                cluster_data = self.data_matrix[cluster_mask]
                cluster_centroid = np.mean(cluster_data, axis=0)
                centroid_img = self.masker.inverse_transform(cluster_centroid)

                # Save centroid beta map
                centroid_path = op.join(
                    k_output_dir, f"cluster_{cluster_id}_centroid_beta.nii.gz"
                )
                nib.save(centroid_img, centroid_path)

                # Threshold beta map (absolute threshold)
                beta_thresh_value = 0.2
                beta_thresh_img = thresh_img(centroid_img, beta_thresh_value)
                beta_thresh_path = op.join(
                    k_output_dir,
                    f"cluster_{cluster_id}_beta_thresh-{beta_thresh_value}.nii.gz",
                )
                nib.save(beta_thresh_img, beta_thresh_path)

                # Convert beta centroid to Z-map
                centroid_data = centroid_img.get_fdata()
                z_data = (centroid_data - self.global_mean) / self.global_std
                z_img = nib.Nifti1Image(z_data, centroid_img.affine)
                z_path = op.join(
                    k_output_dir, f"cluster_{cluster_id}_centroid_zmap.nii.gz"
                )
                nib.save(z_img, z_path)

                # Threshold Z-map (statistical threshold)
                z_thresh_value = 3.09  # ~p<0.001
                z_thresh_img = thresh_img(z_img, z_thresh_value)
                z_thresh_path = op.join(
                    k_output_dir,
                    f"cluster_{cluster_id}_z_thresh-{z_thresh_value}.nii.gz",
                )
                nib.save(z_thresh_img, z_thresh_path)

                # Save participant list
                subjects_file = op.join(
                    k_output_dir, f"cluster_{cluster_id}_participants.txt"
                )
                with open(subjects_file, "w") as f:
                    for subj in cluster_subjects:
                        f.write(f"{subj}\n")

            # Save cluster assignments
            assignments_df = pd.DataFrame(
                {"subject_id": self.subject_ids, "cluster": labels}
            )
            assignments_file = op.join(k_output_dir, f"cluster_assignments_k_{k}.csv")
            assignments_df.to_csv(assignments_file, index=False)

            print(
                f"    Silhouette score: {self.clustering_results[k]['silhouette']:.3f}"
            )

        return self

    def visualize_clusters(self, k_values):
        """Visualize clustering results"""

        print("Creating cluster visualizations...")

        for k in k_values:
            if k not in self.clustering_results:
                continue

            labels = self.clustering_results[k]["labels"]

            # Create cluster visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. PCA visualization
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(self.data_matrix)

            scatter = axes[0, 0].scatter(
                data_2d[:, 0], data_2d[:, 1], c=labels, cmap="tab10", alpha=0.7
            )
            axes[0, 0].set_xlabel(
                f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)"
            )
            axes[0, 0].set_ylabel(
                f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
            )
            axes[0, 0].set_title(f"Clusters in PCA Space (k={k})")
            plt.colorbar(scatter, ax=axes[0, 0])

            # 2. Cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            axes[0, 1].bar(unique, counts, alpha=0.7)
            axes[0, 1].set_xlabel("Cluster ID")
            axes[0, 1].set_ylabel("Number of Participants")
            axes[0, 1].set_title(f"Cluster Sizes (k={k})")

            # 3. Similarity matrix with cluster ordering
            # Reorder similarity matrix by cluster
            cluster_order = np.argsort(labels)
            ordered_similarity = self.similarity_matrix[
                np.ix_(cluster_order, cluster_order)
            ]
            ordered_labels = labels[cluster_order]

            im = axes[1, 0].imshow(
                ordered_similarity, cmap="RdBu_r", vmin=-0.4, vmax=0.4
            )
            axes[1, 0].set_title(f"Similarity Matrix (Ordered by Clusters, k={k})")
            axes[1, 0].set_xlabel("Participant (Ordered)")
            axes[1, 0].set_ylabel("Participant (Ordered)")
            plt.colorbar(im, ax=axes[1, 0])

            # Add cluster boundaries
            cluster_boundaries = []
            current_cluster = ordered_labels[0]
            for i, cluster in enumerate(ordered_labels[1:], 1):
                if cluster != current_cluster:
                    cluster_boundaries.append(i - 0.5)
                    current_cluster = cluster

            for boundary in cluster_boundaries:
                axes[1, 0].axhline(boundary, color="white", linewidth=2)
                axes[1, 0].axvline(boundary, color="white", linewidth=2)

            # 4. Within-cluster similarity distribution
            within_cluster_sims = []
            between_cluster_sims = []

            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if labels[i] == labels[j]:
                        within_cluster_sims.append(self.similarity_matrix[i, j])
                    else:
                        between_cluster_sims.append(self.similarity_matrix[i, j])

            axes[1, 1].hist(
                within_cluster_sims,
                bins=30,
                alpha=0.7,
                label="Within cluster",
                density=True,
            )
            axes[1, 1].hist(
                between_cluster_sims,
                bins=30,
                alpha=0.7,
                label="Between cluster",
                density=True,
            )
            axes[1, 1].set_xlabel("Similarity")
            axes[1, 1].set_ylabel("Density")
            axes[1, 1].set_title(f"Similarity Distributions (k={k})")
            axes[1, 1].legend()

            plt.tight_layout()
            plt.savefig(
                op.join(self.output_dir, f"clusters_k_{k}_visualization.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()

        return self

    def summary_report(self):
        """Generate a summary report of clustering results"""

        print("\n" + "=" * 60)
        print("HABENULA CONNECTIVITY CLUSTERING SUMMARY")
        print("=" * 60)

        print(f"Dataset: {len(self.subject_ids)} participants")
        print(f"Data matrix shape: {self.data_matrix.shape}")
        print(f"Output directory: {self.output_dir}")

        if hasattr(self, "clustering_results"):
            print("\nClustering Results:")
            for k in sorted(self.clustering_results.keys()):
                result = self.clustering_results[k]
                print(f"  k={k}: Silhouette = {result['silhouette']:.3f}")

                # Cluster sizes
                unique, counts = np.unique(result["labels"], return_counts=True)
                sizes_str = ", ".join(
                    [f"C{i}:{count}" for i, count in zip(unique, counts)]
                )
                print(f"        Sizes: {sizes_str}")

        print(f"\nFiles generated in: {self.output_dir}")
        print("  - Similarity visualizations")
        print("  - Cluster validation plots")
        print("  - Individual cluster results (centroids, participant lists)")
        print("  - Cluster assignment files")


def main():
    args = get_parser().parse_args()

    project_dir = args.project_dir
    base_rsfc_dir = args.data_dir
    output_dir = args.out_dir
    method = args.method

    # Ensure output root exists
    os.makedirs(output_dir, exist_ok=True)

    # Metadata path inside project_dir (adjust if yours lives elsewhere)
    metadata_file = op.join(
        project_dir,
        "derivatives",
        "sub-group_task-rest_desc-1S2StTesthabenula_conntable.txt",
    )

    # Initialize clustering object with the two original init args
    clustering = HabenulaClustering(
        project_dir=project_dir, base_rsfc_dir=base_rsfc_dir
    )

    # Set up output directory structure
    clustering.setup_output_directories(output_dir)

    k_range = range(2, 9)

    # Load data and prepare for clustering (common steps)
    clustering.load_data(metadata_file, group_filter="asd")
    clustering.prepare_data_matrix(standardize=True)
    clustering.compute_similarity_matrix()

    # Only create similarity visualizations once (for "all" method or first method)
    if method == "all" or method == "kmeans_no_pca":
        clustering.visualize_similarity()

    if method == "all":
        # Run complete analysis pipeline with all three clustering methods
        (
            clustering.perform_all_clustering_methods(k_values=k_range)
            .visualize_all_clustering_methods(k_values=k_range)
            .compare_clustering_methods(k_values=k_range)
            .summary_report()
        )
    else:
        # Run single clustering method
        clustering.perform_single_clustering_method(method, k_values=k_range)
        print(f"\nCompleted {method} clustering analysis.")
        print(f"Results saved in: {output_dir}/{method}")


if __name__ == "__main__":
    main()
