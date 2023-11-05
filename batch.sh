#!/bin/bash

reduction_methods="umap"
clustering_methods="hdbscan"
size_of_cluster=(3 10 50 100 1000)
for feature_count in {2..10..2}; do
    for cluster_count in $size_of_cluster; do
      for reduction_method in $reduction_methods; do
        for clustering_method in $clustering_methods; do
          # Call the SLURM script for each month and year
          sbatch run_pipeline.sh $feature_count $reduction_method $cluster_count $clustering_method 
        done
      done
    done
done
