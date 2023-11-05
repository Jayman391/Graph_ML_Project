import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

i = params.n_features
dim_red = params.dim_reduction_method
j = params.n_clusters
clust_alg = params.clustering_method

df = pd.read_csv(f'data/{i}_{dim_red}_{j}_{clust_alg}.csv')

#rename first column to id
df.rename(columns={df.columns[0]: 'id'}, inplace=True)

if clust_alg == 'kmeans':
    df_features = df.drop(['id', 'silhouette', 'db'], axis=1)
    plt.figure(figsize=(20,20))
    g = sns.pairplot(df_features, hue='labels', palette='bright', corner=True, diag_kind='hist')
    plt.savefig(f'images/{i}_{dim_red}_{j}_{clust_alg}_pairplot.png')
    plt.close()

if clust_alg == 'hdbscan':
    df_features = df.drop(['id', 'silhouette', 'db',], axis=1)
    plt.figure(figsize=(20,20))
    g = sns.pairplot(df_features, hue='labels',  palette='bright', corner=True, diag_kind='kde')
    plt.savefig(f'images/{i}_{dim_red}_{j}_{clust_alg}_pairplot.png')
    plt.close()