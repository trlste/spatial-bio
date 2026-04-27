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
cd /net/galaxy/home/koes/pas195/Classes/MachineLearning/spatial-bio/Paul_Hier

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

# ─── Phase 3 Run Matrix ──────────────────────────────────────────────────────
# All four runs share the same Hierarchical recipe:
#   - 5 epochs frozen backbone, then partial unfreeze (layer3+layer4 / late blocks)
#   - Discriminative LR: backbone @ 10x lower than head
#   - EMA weights for eval/checkpoint
#   - Deeper trunk: 512 -> 256 -> 128 -> heads
#   - pos_weight=100 on BCE (drops sampler, uses natural distribution)
#   - Warm restarts every 10 epochs
# Differences:
#   Run A: ResNet18,        BCE+CE
#   Run B: EfficientNet-B0, BCE+CE
#   Run A + LibAUC: ResNet18,        pAUC loss (binary head) + 0.1 * type CE
#   Run B + LibAUC: EfficientNet-B0, pAUC loss (binary head) + 0.1 * type CE

# Common flags for all hierarchical runs
COMMON_FLAGS="--model HierarchicalResNet18_smoteenn \
    --hier-freeze-backbone-epochs 20 --hier-partial-unfreeze 1 \
    --hier-pos-weight 100 --hier-trunk-dim-2 128 \
    --backbone-lr-mult 0.1 --use-ema 1 \
    --scheduler cosine_warm_restarts --restart-period 25 \
    --epochs 30 --use-smoteenn 1 --batch-size 128"
#or 2048
#Run A — ResNet18 baseline
#CUDA_VISIBLE_DEVICES=0 python -m src.training $COMMON_FLAGS \
#    --backbone resnet18 --phase-suffix runA_so2 --hier-type-loss-weight 0.5

 #Run B — EfficientNet-B0 baseline
#CUDA_VISIBLE_DEVICES=0 python -m src.training $COMMON_FLAGS \
#    --backbone efficientnet_b0 --phase-suffix runB

# Run A + LibAUC — ResNet18 with pAUC loss
#CUDA_VISIBLE_DEVICES=0 python -m src.training $COMMON_FLAGS \
#    --backbone resnet18 --use-libauc 1 --phase-suffix runA_libauc_longer

# Run B + LibAUC — EfficientNet-B0 with pAUC loss
#CUDA_VISIBLE_DEVICES=0 python -m src.training $COMMON_FLAGS \
#    --backbone efficientnet_b0 --use-libauc 1 --phase-suffix runB_libauc

# Run 1: two-stage at unfreeze boundary (stage2_epoch == freeze_epochs == 20)
#CUDA_VISIBLE_DEVICES=0 python -m src.training $COMMON_FLAGS \
#    --backbone resnet18 \
#    --libauc-mode two_stage --libauc-stage2-epoch 20 --libauc-num-pos 64 \
#    --phase-suffix runA_2stage20

# Run 2: earlier stage-2 entry, spans frozen->unfrozen
CUDA_VISIBLE_DEVICES=0 python -m src.training $COMMON_FLAGS \
    --backbone resnet18 \
    --libauc-mode two_stage --libauc-stage2-epoch 10 --libauc-num-pos 64 \
    --phase-suffix runA_2stage10 --hier-type-loss-weight 0.5

# Run 3 (sanity baseline): SMOTE + full LibAUC, previously blocked
#CUDA_VISIBLE_DEVICES=0 python -m src.training $COMMON_FLAGS \
#    --backbone resnet18 \
#    --libauc-mode full --libauc-num-pos 64 \
#    --phase-suffix runA_smote_libauc_full