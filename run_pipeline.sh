#!/bin/bash
#SBATCH --partition=bluemoon
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --time=10:00:00
#SBATCH --mem=128G
#SBATCH --job-name=GRAPH_ML_USER_FEATURE_ANALYSIS
#SBATCH --output=output/%x_%j.out
#SBATCH --mail-type=FAIL

python pipeline.py