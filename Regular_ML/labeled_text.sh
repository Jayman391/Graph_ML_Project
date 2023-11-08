#!/bin/bash
#SBATCH --partition=bluemoon
#SBATCH --nodes=8
#SBATCH --ntask=8
#SBATCH --time=20:00:00
#SBATCH --mem=512G
#SBATCH --job-name=GRAPH_ML_USER_FEATURE_ANALYSIS
#SBATCH --output=output/%x_%j.out
#SBATCH --mail-type=FAIL

python add_comments.py