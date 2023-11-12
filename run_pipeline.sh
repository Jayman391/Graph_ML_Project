#!/bin/sh
#SBATCH --partition=dggpu
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --time=25:00:00
#SBATCH --mem=256G
#SBATCH --job-name=full_gml_pipeline
#SBATCH --output=output/%x_%j.out
#SBATCH --mail-type=FAIL

python pipeline.py
