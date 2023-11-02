#!/bin/bash

reduction_methods="pca umap"
clustering_methods="kmeans hdbscan"

for feature_count in {3..5}; do
    for cluster_count in {3..10}; do
      for reduction_method in $reduction_methods; do
        for clustering_method in $clustering_methods; do
          # Call the SLURM script for each month and year
          sbatch run_pairplots.sh $feature_count $reduction_method $cluster_count $clustering_method 
        done
      done
    done
done
