#!/bin/sh
#SBATCH --time=1-12
#SBATCH --mem=30g
#SBATCH --gres=gpu:1

source activate venv

python embeddings_analysis.py