# src/training.py

import os
import argparse
from Model.attention_resnet18 import AttentionResNet18
from Model.naive_resnet18 import NaiveResNet18
from src.train_one_fold import train_one_fold

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    choices=['AttentionResNet18', 'NaiveResNet18'])
args = parser.parse_args()

base_config = {
    'csv_path'     : 'data/ISIC_2024_Training_Supplement_processed.csv',
    'image_dir'    : 'data/ISIC_2024_Training_Input',
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
}

model_map = {
        'AttentionResNet18': AttentionResNet18,
        'NaiveResNet18'    : NaiveResNet18,
    }

model_cls = model_map[args.model]

for fold in range(base_config['n_folds']):
    model  = model_cls(n_classes=base_config['n_classes'],
                        dropout=base_config['dropout'],
                        pretrained=True)
    config = {**base_config, 'model_name': args.model}
    print(f"\n{'='*50}")
    print(f"Training {args.model} | Fold {fold}")
    print(f"{'='*50}")
    best_auc = train_one_fold(model, fold, config)
    print(f"Best AUC: {best_auc:.4f}")