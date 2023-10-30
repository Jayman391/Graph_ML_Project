import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse


# load in features and preprocess data

def preprocess_data():
  # load in data
  groups = pd.read_json('babynamesDB_groups.json')
  # filter out groups with less than 3 users
  groups = groups[groups['num_users_stored'] > 3]
  group_ids = groups['_id'].to_list()

  print('group data successfully loaded')

  users = pd.read_json('babynamesDB_users.json')
  users = users[['_id' , 'num_comments_stored', 'groups', 'num_posts_stored']]
  # unnest groups
  users = users.explode('groups')
  users = users[users['groups'].isin(group_ids)]
  # 1 hot encode groups
  users = pd.concat([users, pd.get_dummies(users['groups'], dtype=float)], axis=1)
  # join data on user id
  users = users.groupby('_id').sum()
  users = users.drop(columns=['groups'])

  print('user data successfully loaded')

  # scale data
  scaler = StandardScaler()
  users_scaled = scaler.fit_transform(users)

  print('data successfully scaled')

  return users_scaled

def apply_dimensionality_reduction(data, n_components, method):

  method = method.upper()

  if method == 'PCA':
    model = PCA(n_components=n_components)
  elif method == 'UMAP':
    model = umap.UMAP(n_components=n_components)
  else:
    raise ValueError('Invalid dimensionality reduction method')
  
  data_reduced = model.fit_transform(data)

  print(f'data successfully reduced using {method} to {n_components} dimensions')

  return data_reduced


def apply_unsupervised_clustering(data, n_clusters, method):
  
  method = method.upper()

  if method == 'KMEANS':
    model = KMeans(n_clusters=n_clusters)
  elif method == 'HDBSCAN':
    model = hdbscan.HDBSCAN(min_cluster_size=n_clusters)
  else:
    raise ValueError('Invalid clustering method')
  
  labels = model.fit_predict(data)

  print(f'data successfully clustered using {method} into {n_clusters} clusters')

  return labels


def evaluate_unsupervised_labels(data, labels):
  
    # calculate silhouette score
    silhouette = silhouette_score(data, labels)
    # calculate davies bouldin score
    db = davies_bouldin_score(data, labels)
    # calculate calinski harabasz score
    ch = calinski_harabasz_score(data, labels)
  
    return silhouette, db, ch


def composite_score(silhouette, db, ch):
  ''' 
  the closer the davies bouldin score is to 0 the better
  the support of the silhouette score is between -1 and 1
  the higher the calinski harabasz score the better

  This composition turns the score into a maximization problem
  '''
  return (1-db) * silhouette * ch


def plot_clusters(data_reduced, labels, dim_reduction_method, clustering_method, n_features, n_clusters, scores):
    # Convert the data to a DataFrame
    df = pd.DataFrame(data_reduced, columns=[f'Feature_{i}' for i in range(1, n_features+1)])
    df['Cluster'] = labels
    
    # Output statistics about features
    print("\nFeature Statistics:")
    print(df.describe(include=[np.number]))
    
    # Output statistics about clusters
    print("\nCluster Counts:")
    print(df['Cluster'].value_counts())
    
    # Create pairplot colored by cluster
    pairplot = sns.pairplot(df, hue='Cluster', palette='Set2')
    plt.title(f'Pairplot of Features Colored by Cluster (Dim Reduction: {dim_reduction_method}, Clustering: {clustering_method}, n_features: {n_features}, n_clusters: {n_clusters})', y=1.02)
    plt.suptitle(f'Silhouette Score: {scores[0]}, Davies-Bouldin Score: {scores[1]}, Calinski-Harabasz Score: {scores[2]}', y=0.95)

    # Show the plot
    plt.show()

    # Save the plot
    pairplot.savefig(f'output/pairplot_{dim_reduction_method}_{clustering_method}_{n_features}features_{n_clusters}clusters.png')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run unsupervised learning on baby names dataset.')
    parser.add_argument('--n_features', type=int, required=True, help='Number of features for dimensionality reduction')
    parser.add_argument('--dim_reduction_method', type=str, required=True, choices=['pca', 'umap'], help='Dimensionality reduction method')
    parser.add_argument('--n_clusters', type=int, required=True, help='Number of clusters for clustering algorithm')
    parser.add_argument('--clustering_method', type=str, required=True, choices=['kmeans', 'hdbscan'], help='Clustering algorithm')
    return parser.parse_args()

def run_analysis(data, params):
    # Apply dimensionality reduction
    data_reduced = apply_dimensionality_reduction(data, params['n_features'], params['dim_reduction_method'])
    
    # Print statistics about the reduced features
    print(f'Mean of reduced features: {np.mean(data_reduced, axis=0)}')
    print(f'Standard deviation of reduced features: {np.std(data_reduced, axis=0)}')
    
    # Apply clustering
    labels = apply_unsupervised_clustering(data_reduced, params['n_clusters'], params['clustering_method'])
    
    # Evaluate clustering
    silhouette, db, ch = evaluate_unsupervised_labels(data_reduced, labels)
    scores = [silhouette, db, ch]
    print(f'Silhouette Score: {silhouette}')
    print(f'Davies-Bouldin Score: {db}')
    print(f'Calinski-Harabasz Score: {ch}')
    
    # Plot labeled clusters
    plot_clusters(data_reduced, labels, params['dim_reduction_method'], params['clustering_method'], params['n_features'], params['n_clusters'], scores)


if __name__ == '__main__':

  data = preprocess_data()

  params = parse_arguments()

  run_analysis(data, params)

