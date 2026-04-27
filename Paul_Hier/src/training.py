# src/training.py

import os
import argparse
from pathlib import Path
from Model.attention_resnet18 import AttentionResNet18
from Model.naive_resnet18 import NaiveResNet18
from Model.custom_cnn import CustomCNN
from Model.custom_cnn_residual_attention import CustomCNNResidualAttention
from Model.hierarchical_resnet18 import HierarchicalResNet18
from Model.efficient_net_b0 import EfficientNetB0, HierarchicalEfficientNetB0
from src.train_one_fold import train_one_fold

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / 'data'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    choices=['AttentionResNet18', 'NaiveResNet18', 'CustomCNN',
                             'CustomCNNResidualAttention', 'HierarchicalResNet18',
                             'NaiveResNet18_smoteenn', 'HierarchicalResNet18_smoteenn'])

# Custom-CNN-specific knobs (unchanged)
parser.add_argument('--custom-base-channels', type=int, default=64)
parser.add_argument('--custom-adaptive-pool-size', type=int, default=8)
parser.add_argument('--custom-classifier-hidden-dim', type=int, default=1024)
parser.add_argument('--custom-classifier-bottleneck-dim', type=int, default=256)
parser.add_argument('--custom-ra-residual-depth', type=int, default=2)
parser.add_argument('--custom-ra-use-attention', type=int, choices=[0, 1], default=1)
parser.add_argument('--custom-ra-attention-skip', type=int, default=1)

# Hierarchical-specific knobs
parser.add_argument('--hier-alpha', type=float, default=1.0,
                    help='Weight on binary loss in hierarchical model.')
parser.add_argument('--hier-beta', type=float, default=1.0,
                    help='Weight on type loss in hierarchical model.')
parser.add_argument('--hier-trunk-dim', type=int, default=256,
                    help='Width of shared trunk between backbone and heads.')
parser.add_argument('--hier-trunk-dim-2', type=int, default=0,
                    help='Width of optional second trunk layer; 0 = single-layer trunk.')
parser.add_argument('--hier-freeze-backbone-epochs', type=int, default=0,
                    help='Freeze backbone for this many initial epochs (0 = never).')
parser.add_argument('--hier-partial-unfreeze', type=int, choices=[0, 1], default=0,
                    help='At unfreeze time, only unfreeze late stages (1) or full backbone (0).')
parser.add_argument('--hier-pos-weight', type=float, default=100.0,
                    help='pos_weight on BCE for malignant class in hierarchical loss.')
parser.add_argument('--hier-type-loss-weight', type=float, default=0.1,
                    help='Auxiliary type CE weight when using libauc (default 0.1).')

# Backbone selector
parser.add_argument('--backbone', choices=['resnet18', 'efficientnet_b0'], default='resnet18',
                    help='Backbone for NaiveResNet18 / NaiveResNet18_smoteenn / HierarchicalResNet18 / HierarchicalResNet18_smoteenn.')

# Training-recipe toggles
parser.add_argument('--use-sampler', type=int, choices=[0, 1], default=0,
                    help='Enable WeightedRandomSampler with sqrt class weights.')
parser.add_argument('--use-smoteenn', type=int, choices=[0, 1], default=None,
                    help='Enable SMOTE warmup explicitly (1=on, 0=off). If omitted, inferred from *_smoteenn model names for backward compatibility.')
parser.add_argument('--libauc-mode', choices=['off', 'full', 'two_stage'], default='off',
                    help='LibAUC training mode. off=HierarchicalLoss only (default). '
                         'full=PAUCLoss + SOPAs + DualSampler from epoch 0. '
                         'two_stage=HierarchicalLoss for epochs [0, stage2_epoch), '
                         'then switch to PAUCLoss + SOPAs + DualSampler for the rest. '
                         'Replaces the old --use-libauc flag.')
parser.add_argument('--libauc-stage2-epoch', type=int, default=-1,
                    help='Epoch at which two_stage mode switches from CE/BCE to pAUC. '
                         'Required when --libauc-mode=two_stage; must be in (0, epochs). '
                         'Ignored (and must be -1) for other modes.')
parser.add_argument('--libauc-sampling-rate', type=float, default=0.1,
                    help='DualSampler positive sampling rate for LibAUC runs (ignored if --libauc-num-pos > 0).')
parser.add_argument('--libauc-num-pos', type=int, default=0,
                    help='Exact positives per batch for LibAUC DualSampler (0 = derive from sampling rate).')
parser.add_argument('--use-ema', type=int, choices=[0, 1], default=0,
                    help='Apply EMA to model weights for evaluation/checkpoint.')
parser.add_argument('--ema-decay', type=float, default=0.999,
                    help='EMA decay rate (closer to 1 = slower averaging).')
parser.add_argument('--backbone-lr-mult', type=float, default=0.1,
                    help='Discriminative LR: backbone LR = lr * this multiplier.')
parser.add_argument('--scheduler', choices=['cosine', 'cosine_warm_restarts'], default='cosine',
                    help='LR scheduler type.')
parser.add_argument('--restart-period', type=int, default=10,
                    help='T_0 for CosineAnnealingWarmRestarts (epochs per cycle).')
parser.add_argument('--smote-warmup-epochs', type=int, default=5,
                    help='Warmup epochs for SMOTE head pretraining.')
parser.add_argument('--smote-head-lr', type=float, default=1e-3,
                    help='Head learning rate used during SMOTE warmup.')

parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight-decay', type=float, default=1e-2)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--phase-suffix', type=str, default='',
                    help='Optional suffix appended to auto-generated phase tag.')

args = parser.parse_args()


# Validate libauc-mode / libauc-stage2-epoch combination.
if args.libauc_mode == 'two_stage':
    if args.libauc_stage2_epoch <= 0:
        raise ValueError(
            '--libauc-mode two_stage requires --libauc-stage2-epoch > 0.'
        )
    if args.libauc_stage2_epoch >= args.epochs:
        raise ValueError(
            f'--libauc-stage2-epoch ({args.libauc_stage2_epoch}) must be '
            f'< --epochs ({args.epochs}); otherwise the switch never fires.'
        )
    if args.libauc_stage2_epoch < args.hier_freeze_backbone_epochs:
        print(
            f'[warn] --libauc-stage2-epoch ({args.libauc_stage2_epoch}) is < '
            f'--hier-freeze-backbone-epochs ({args.hier_freeze_backbone_epochs}); '
            f'pAUC training will start while the backbone is still frozen.'
        )
elif args.libauc_stage2_epoch != -1:
    raise ValueError(
        f'--libauc-stage2-epoch is only valid with --libauc-mode two_stage '
        f'(got mode={args.libauc_mode!r}).'
    )


def _resolve_use_smoteenn(args):
    model_implies_smote = args.model in ('NaiveResNet18_smoteenn', 'HierarchicalResNet18_smoteenn')
    if args.use_smoteenn is None:
        return model_implies_smote
    return bool(args.use_smoteenn)


USE_SMOTEENN = _resolve_use_smoteenn(args)

if USE_SMOTEENN and args.model not in (
    'NaiveResNet18',
    'NaiveResNet18_smoteenn',
    'HierarchicalResNet18',
    'HierarchicalResNet18_smoteenn',
):
    raise ValueError(
        '--use-smoteenn is only supported for NaiveResNet18 and HierarchicalResNet18 variants.'
    )


# Auto-generate a descriptive phase tag from the active flags so each W&B
# run is identifiable without manual bookkeeping.
def _phase_tag(args):
    parts = []
    if args.model in ('HierarchicalResNet18', 'HierarchicalResNet18_smoteenn'):
        parts.append('hier')
    else:
        parts.append('naive')
    parts.append(args.backbone)  # resnet18 or efficientnet_b0
    if USE_SMOTEENN:
        parts.append('smoteenn')
    if args.use_sampler:
        parts.append('sampler')
    if args.libauc_mode == 'full':
        parts.append('libauc')
    elif args.libauc_mode == 'two_stage':
        parts.append(f'libauc2s{args.libauc_stage2_epoch}')
    if args.use_ema:
        parts.append('ema')
    if args.hier_partial_unfreeze and args.model in ('HierarchicalResNet18', 'HierarchicalResNet18_smoteenn'):
        parts.append('partunfreeze')
    if args.scheduler == 'cosine_warm_restarts':
        parts.append('warmrestart')
    if args.phase_suffix:
        parts.append(args.phase_suffix)
    return '_'.join(parts)


base_config = {
    'csv_path'     : str(DATA_ROOT / 'ISIC_2024_Training_Supplement_processed.csv'),
    'image_dir'    : str(DATA_ROOT / 'ISIC_2024_Training_Input'),
    'img_size'     : 112,
    'batch_size'   : args.batch_size,
    'epochs'       : args.epochs,
    'lr'           : args.lr,
    'weight_decay' : args.weight_decay,
    'dropout'      : 0.3,
    'n_folds'      : 3,
    'n_classes'    : 5,
    'aug_prob'     : 1.0,
    'seed'         : 42,
    'phase'        : _phase_tag(args),
    'backbone'     : args.backbone,

    # custom-cnn knobs
    'custom_base_channels'             : args.custom_base_channels,
    'custom_adaptive_pool_size'        : args.custom_adaptive_pool_size,
    'custom_classifier_hidden_dim'     : args.custom_classifier_hidden_dim,
    'custom_classifier_bottleneck_dim' : args.custom_classifier_bottleneck_dim,
    'custom_ra_residual_depth'         : args.custom_ra_residual_depth,
    'custom_ra_use_attention'          : bool(args.custom_ra_use_attention),
    'custom_ra_attention_skip'         : args.custom_ra_attention_skip,

    # hierarchical knobs
    'hier_alpha'                       : args.hier_alpha,
    'hier_beta'                        : args.hier_beta,
    'hier_trunk_dim'                   : args.hier_trunk_dim,
    'hier_trunk_dim_2'                 : args.hier_trunk_dim_2,
    'hier_freeze_backbone_epochs'      : args.hier_freeze_backbone_epochs,
    'hier_partial_unfreeze'            : bool(args.hier_partial_unfreeze),
    'hier_pos_weight'                  : args.hier_pos_weight,
    'hier_type_loss_weight'            : args.hier_type_loss_weight,

    # training-recipe toggles
    'use_sampler'                      : bool(args.use_sampler),
    'use_smoteenn'                     : USE_SMOTEENN,
    'libauc_mode'                      : args.libauc_mode,
    'libauc_stage2_epoch'              : args.libauc_stage2_epoch,
    'libauc_sampling_rate'             : args.libauc_sampling_rate,
    'libauc_num_pos'                   : (args.libauc_num_pos if args.libauc_num_pos > 0 else None),
    'use_ema'                          : bool(args.use_ema),
    'ema_decay'                        : args.ema_decay,
    'smote_warmup_epochs'              : args.smote_warmup_epochs,
    'smote_head_lr'                    : args.smote_head_lr,
    'backbone_lr_mult'                 : args.backbone_lr_mult,
    'scheduler'                        : args.scheduler,
    'restart_period'                   : args.restart_period,
}


# Pick the right model class based on (model_arch, backbone).
def _select_model_class(model_arg, backbone_arg):
    if model_arg in ('NaiveResNet18', 'NaiveResNet18_smoteenn'):
        return EfficientNetB0 if backbone_arg == 'efficientnet_b0' else NaiveResNet18
    if model_arg in ('HierarchicalResNet18', 'HierarchicalResNet18_smoteenn'):
        return HierarchicalEfficientNetB0 if backbone_arg == 'efficientnet_b0' else HierarchicalResNet18
    # The rest don't have backbone variants.
    fixed = {
        'AttentionResNet18'         : AttentionResNet18,
        'CustomCNN'                 : CustomCNN,
        'CustomCNNResidualAttention': CustomCNNResidualAttention,
    }
    return fixed[model_arg]


model_cls = _select_model_class(args.model, args.backbone)

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
    elif args.model in ('HierarchicalResNet18', 'HierarchicalResNet18_smoteenn'):
        # model_cls is HierarchicalResNet18 OR HierarchicalEfficientNetB0
        model = model_cls(
            n_classes=base_config['n_classes'],
            dropout=base_config['dropout'],
            pretrained=True,
            trunk_dim=base_config['hier_trunk_dim'],
            trunk_dim_2=base_config['hier_trunk_dim_2'],
        )
    else:
        # NaiveResNet18 or EfficientNetB0 — same constructor signature.
        model = model_cls(
            n_classes=base_config['n_classes'],
            dropout=base_config['dropout'],
            pretrained=True,
        )

    config = {**base_config, 'model_name': args.model}
    print(f"\n{'='*50}")
    print(f"Training {args.model} ({args.backbone}) | phase={config['phase']} | Fold {fold}")
    print(f"{'='*50}")
    best_pauc = train_one_fold(model, fold, config)
    print(f"Best pAUC (TPR>=80%): {best_pauc:.4f}")