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
df.rename(columns={'Unnamed: 0':'id'}, inplace=True)
df_features = df.drop(['id', 'silhouette', 'db', 'ch', 'composite_score'], axis=1)
plt.figure()
sns.pairplot(df_features, corner=True, hue='labels')
plt.title(f'Pairplot of Features for {i} dimensions {dim_red} {j} clusters {clust_alg}')
plt.savefig(f'images/{i}_{dim_red}_{j}_{clust_alg}_pairplot.png')
plt.close()
