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


def grid_search(data, param_grid):

  best_score = -np.inf
  best_params = None
  best_labels = None

  for params in ParameterGrid(param_grid):
    # apply dimensionality reduction
    data_reduced = apply_dimensionality_reduction(data, params['n_features'], params['dim_reduction_method'])
    # apply clustering
    labels = apply_unsupervised_clustering(data_reduced, params['n_clusters'], params['clustering_method'])
    # evaluate clustering
    silhouette, db, ch = evaluate_unsupervised_labels(data_reduced, labels)
    # update best score
    score = composite_score(silhouette, db, ch)
    scores = [silhouette, db, ch]
     # plot labeled clusters
    plot_clusters(data_reduced, labels, params['dim_reduction_method'], params['clustering_method'], params['n_features'], params['n_clusters'], scores)

    if score > best_score:
      best_score = score
      best_params = params
      best_labels = labels
  

  return best_score, best_params, best_labels


def plot_clusters(data, labels, dim_red_technique, clustering_technique, n_features, n_clusters, scores):
    # Convert the input data and labels into a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Add the cluster labels as a new column in the DataFrame
    df['Cluster'] = labels

    
    
    # Create a pair plot with colors representing clusters
    pair_plot = sns.pairplot(df, hue='Cluster', palette='viridis')
    
    plt.title(f"{dim_red_technique} with {clustering_technique} on {n_features} features and {n_clusters} clusters")
    plt.suptitle(f"Silhouette: {scores[0]:.2f}, Davies-Bouldin: {scores[1]:.2f}, Calinski-Harabasz: {scores[2]:.2f}")
    # Show the plot
    plt.show()
    
    # Save the plot to a PNG file
    filename = f"graphs/{dim_red_technique}_{clustering_technique}_{n_features}features_{n_clusters}clusters.png"
    pair_plot.savefig(filename)
    print(f"Plot saved as {filename}")

if __name__ == '__main__':

  data = preprocess_data()

  param_grid = {
        'n_features': list(range(2,11)),
        'dim_reduction_method': ['pca', 'umap'],
        'n_clusters': list(range(2,11)),
        'clustering_method': ['kmeans', 'hdbscan']
    }

  if not os.path.exists('output'):
    os.makedirs('output')

  best_score, best_params, best_labels = grid_search(data, param_grid)

  print(f'Best score: {best_score}')
  print(f'Best params: {best_params}')
  print(f'Best labels: {best_labels}')