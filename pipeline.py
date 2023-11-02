import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.model_selection import ParameterGrid
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# load in features and preprocess data




def apply_dimensionality_reduction(data, n_components, method):
    method = method.upper()

    if method == "PCA":
        model = PCA(n_components=n_components)
    elif method == "UMAP":
        model = umap.UMAP(n_components=n_components)
    else:
        raise ValueError("Invalid dimensionality reduction method")

    data_reduced = model.fit_transform(data)

    print(f"data successfully reduced using {method} to {n_components} dimensions")

    return data_reduced


def apply_unsupervised_clustering(data, n_clusters, method):
    method = method.upper()

    if method == "KMEANS":
        model = KMeans(n_clusters=n_clusters)
    elif method == "HDBSCAN":
        model = hdbscan.HDBSCAN(min_cluster_size=len(data)//(n_clusters * 2))
    else:
        raise ValueError("Invalid clustering method")

    labels = model.fit_predict(data)

    print(f"data successfully clustered using {method} into {n_clusters} clusters")

    return labels


def evaluate_unsupervised_labels(data, labels):
    # calculate silhouette score
    silhouette = silhouette_score(data, labels)
    # calculate davies bouldin score
    db = davies_bouldin_score(data, labels)
    # calculate calinski harabasz score

    return silhouette, db



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run unsupervised learning on baby names dataset."
    )
    parser.add_argument(
        "n_features", type=int, help="Number of features for dimensionality reduction"
    )
    parser.add_argument(
        "dim_reduction_method",
        type=str,
        choices=["pca", "umap"],
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "n_clusters", type=int, help="Number of clusters for clustering algorithm"
    )
    parser.add_argument(
        "clustering_method",
        type=str,
        choices=["kmeans", "hdbscan"],
        help="Clustering algorithm",
    )
    return parser.parse_args()


def run_analysis(data, params):
    # Apply dimensionality reduction
    data_reduced = apply_dimensionality_reduction(
        data, params.n_features, params.dim_reduction_method
    )
    # Print statistics about the reduced features
    print(f"Mean of reduced features: {np.mean(data_reduced, axis=0)}")
    print(f"Standard deviation of reduced features: {np.std(data_reduced, axis=0)}")

    # Apply clustering
    labels = apply_unsupervised_clustering(
        data_reduced, params.n_clusters, params.clustering_method
    )
    # Evaluate clustering
    silhouette, db= evaluate_unsupervised_labels(data_reduced, labels)
    scores = [silhouette, db]
    print(f"Silhouette Score: {silhouette}")
    print(f"Davies-Bouldin Score: {db}")

    # write everything to a csv
    df = pd.DataFrame(data_reduced)
    df["labels"] = labels
    df["silhouette"] = silhouette
    df["db"] = db
    df.to_csv(f'data/{params.n_features}_{params.dim_reduction_method}_{params.n_clusters}_{params.clustering_method}.csv')


if __name__ == "__main__":
    data = pd.read_csv("data/users_embeddings.csv")

    params = parse_arguments()

    run_analysis(data, params)
