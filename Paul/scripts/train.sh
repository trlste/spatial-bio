#!/bin/bash
#SBATCH --job-name=isic_training
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --partition=koes_gpu
#SBATCH --constraint=L40
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# ─── environment setup ──────────────────────────────────────────────────────
echo ""
echo "Setting up conda environment..."
source /net/dali/home/mscbio/pas195/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate isic2024

echo "Activated conda environment: $(conda info --envs | grep '*')"
echo "Python path: $(which python)"
echo ""
cd /net/galaxy/home/koes/pas195/Classes/MachineLearning/spatial-bio/Paul

# ─── verify PyTorch and GPU ─────────────────────────────────────────────────
echo "Verifying PyTorch and GPU setup..."
python << 'PYCHECK'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA not available!")
    exit(1)
PYCHECK

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch/CUDA verification failed!"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv


echo ""
echo "========================================="


mkdir -p logs checkpoints

#CUDA_VISIBLE_DEVICES=0 python -m src.training --model CustomCNN

#CUDA_VISIBLE_DEVICES=0 python -m src.training --model CustomCNNResidualAttention

#CUDA_VISIBLE_DEVICES=0 python -m src.training --model AttentionResNet18

#CUDA_VISIBLE_DEVICES=0 python -m src.training --model NaiveResNet18

#wait  # important -- keeps the job alive until both processes finish