# src/training.py

import os
import argparse
from pathlib import Path
from Model.attention_resnet18 import AttentionResNet18
from Model.naive_resnet18 import NaiveResNet18
from Model.custom_cnn import CustomCNN
from Model.custom_cnn_residual_attention import CustomCNNResidualAttention
from src.train_one_fold import train_one_fold

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / 'data'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    choices=['AttentionResNet18', 'NaiveResNet18', 'CustomCNN', 'CustomCNNResidualAttention'])
parser.add_argument('--custom-base-channels', type=int, default=64)
parser.add_argument('--custom-adaptive-pool-size', type=int, default=8)
parser.add_argument('--custom-classifier-hidden-dim', type=int, default=1024)
parser.add_argument('--custom-classifier-bottleneck-dim', type=int, default=256)
parser.add_argument('--custom-ra-residual-depth', type=int, default=2)
parser.add_argument('--custom-ra-use-attention', type=int, choices=[0, 1], default=1)
parser.add_argument('--custom-ra-attention-skip', type=int, default=1)
args = parser.parse_args()

base_config = {
    'csv_path'     : str(DATA_ROOT / 'ISIC_2024_Training_Supplement_processed.csv'),
    'image_dir'    : str(DATA_ROOT / 'ISIC_2024_Training_Input'),
    'img_size'     : 135,
    'batch_size'   : 256,
    'epochs'       : 30,
    'lr'           : 2e-4,
    'weight_decay' : 1e-2,
    'dropout'      : 0.4,
    'n_folds'      : 3,
    'n_classes'    : 5,
    'aug_prob'     : 0.75,
    'seed'         : 42,
    'custom_base_channels'             : args.custom_base_channels,
    'custom_adaptive_pool_size'        : args.custom_adaptive_pool_size,
    'custom_classifier_hidden_dim'     : args.custom_classifier_hidden_dim,
    'custom_classifier_bottleneck_dim' : args.custom_classifier_bottleneck_dim,
    'custom_ra_residual_depth'         : args.custom_ra_residual_depth,
    'custom_ra_use_attention'          : bool(args.custom_ra_use_attention),
    'custom_ra_attention_skip'         : args.custom_ra_attention_skip,
}

model_map = {
        'AttentionResNet18': AttentionResNet18,
        'NaiveResNet18'    : NaiveResNet18,
        'CustomCNN'        : CustomCNN,
        'CustomCNNResidualAttention': CustomCNNResidualAttention,
    }

model_cls = model_map[args.model]

for fold in range(base_config['n_folds']):
    if args.model == 'CustomCNN':
        model = model_cls(
            n_classes=base_config['n_classes'],
            dropout=base_config['dropout'],
            pretrained=False,
            base_channels=base_config['custom_base_channels'],
            adaptive_pool_size=base_config['custom_adaptive_pool_size'],
            classifier_hidden_dim=base_config['custom_classifier_hidden_dim'],
            classifier_bottleneck_dim=base_config['custom_classifier_bottleneck_dim'],
        )
    elif args.model == 'CustomCNNResidualAttention':
        model = model_cls(
            n_classes=base_config['n_classes'],
            dropout=base_config['dropout'],
            pretrained=False,
            base_channels=base_config['custom_base_channels'],
            adaptive_pool_size=base_config['custom_adaptive_pool_size'],
            classifier_hidden_dim=base_config['custom_classifier_hidden_dim'],
            classifier_bottleneck_dim=base_config['custom_classifier_bottleneck_dim'],
            residual_depth=base_config['custom_ra_residual_depth'],
            use_attention=base_config['custom_ra_use_attention'],
            attention_skip=base_config['custom_ra_attention_skip'],
        )
    else:
        model = model_cls(
            n_classes=base_config['n_classes'],
            dropout=base_config['dropout'],
            pretrained=True,
        )
    config = {**base_config, 'model_name': args.model}
    print(f"\n{'='*50}")
    print(f"Training {args.model} | Fold {fold}")
    print(f"{'='*50}")
    best_auc = train_one_fold(model, fold, config)
    print(f"Best AUC: {best_auc:.4f}")