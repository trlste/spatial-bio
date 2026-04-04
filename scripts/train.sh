#!/bin/bash
#SBATCH --job-name=isic_training
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:2

source ~/.bashrc
conda activate isic2024

cd /net/dali/home/mscbio/yur28/spatial-bio

CUDA_VISIBLE_DEVICES=0 python -m src.training --model AttentionResNet18 &
CUDA_VISIBLE_DEVICES=1 python -m src.training --model NaiveResNet18 &

wait  # important -- keeps the job alive until both processes finish