#!/bin/bash
#SBATCH --partition=bluemoon
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --time=20:00:00
#SBATCH --mem=512G
#SBATCH --job-name=GRAPH_ML_PREPROCESS
#SBATCH --output=output/%x_%j.out
#SBATCH --mail-type=FAIL

python preprocess_posts.py
