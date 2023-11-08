#!/bin/bash
#SBATCH --partition=dggpu
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --time=20:00:00
#SBATCH --mem=512G
#SBATCH --job-name=zipf_babycenter_clusters
#SBATCH --output=output/%x_%j.out
#SBATCH --mail-type=FAIL

python pipeline.py "$@"
