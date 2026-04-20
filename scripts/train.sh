#!/bin/bash
#SBATCH --job-name=isic_training
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g011,g006,g013,g[014-015],g123

source ~/.bashrc
conda activate isic2024

cd /net/dali/home/mscbio/yur28/spatial-bio

python -m src.training --model NaiveResNet18