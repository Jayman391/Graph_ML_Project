#!/bin/bash

reduction_methods = "PCA UMAP"
clustering_methods = "KMeans"

for feature_count in {2..10}; do
    for cluster_count in {2..10}; do
      for reduction_method in reduction_methods; do
        # Call the SLURM script for each month and year
        sbatch run_pipeline.sh $feature_count $reduction_method $clustering_methods $cluster_count
    done
done