#!/bin/bash
#SBATCH --partition=bluemoon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --job-name=GRAPH_ML_PREPROCESS
#SBATCH --output=output/%x_%j.out
#SBATCH --mail-type=FAIL

python make_pairplots.py "$@"
