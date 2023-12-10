#!/bin/sh
#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=25:00:00
#SBATCH --mem=512G
#SBATCH --job-name=full_gml_pipeline
#SBATCH --output=output/%x_%j.out
#SBATCH --mail-type=FAIL

python pipeline.py