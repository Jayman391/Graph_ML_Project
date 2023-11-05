import pandas as pd
import argparse 

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

params = parse_arguments()

def get_cluster_size_distributions(params):
    data = pd.read_csv(f"data/feature_clusters/{params.n_features}_{params.dim_reduction_method}_{params.n_clusters}_{params.clustering_method}.csv")
    labels = data['labels']
    cluster_size_distributions = labels.value_counts()
    total_clusters = len(cluster_size_distributions)
    relative_frequencies = cluster_size_distributions / sum(cluster_size_distributions)
    # Create a DataFrame from the calculated data
    zipfian_df = pd.DataFrame({
        'cluster': range(total_clusters),  # Assuming cluster indices start from 0
        'cluster_size': cluster_size_distributions.values,  # The sizes of each cluster
        'total_clusters': total_clusters,  # This will be the same value for all rows, indicating the total number of clusters
        'relative_frequency': relative_frequencies.values  # The relative frequencies
    })
    zipfian_df = pd.DataFrame(zipfian_df, columns=['cluster', 'cluster_size', 'total_clusters', 'relative_frequency'])
    # make cluster index
    zipfian_df.set_index('cluster', inplace=True)
    zipfian_df.to_csv(f'data/size_distributions/{params.n_features}_{params.dim_reduction_method}_{params.n_clusters}_{params.clustering_method}_cluster_size_distributions.csv')

get_cluster_size_distributions(params)