# src/train_one_fold.py

import copy
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from .logger import (
    collect_gradient_stats,
    compute_classification_metrics,
    compute_classification_plots,
    finish_wandb,
    init_wandb,
    log_metrics,
    summarize_gradient_stats,
)
from .dataset import get_datasets, apply_smoteenn
from .transforms import train_transform, val_transform
from .hierarchical_loss import HierarchicalLoss
from Model.hierarchical_resnet18 import HierarchicalResNet18
from Model.efficient_net_b0 import HierarchicalEfficientNetB0


def _resolve_benign_idx(class_names):
    """Find the index of the benign class in the categorical ordering."""
    for i, name in enumerate(class_names):
        if 'benign' in str(name).lower():
            return i
    return 1  # cat.codes alphabetical default: BCC=0, Benign=1, ...


def _is_hierarchical(model):
    """Returns True for any hierarchical (dual-head) model variant."""
    return isinstance(model, (HierarchicalResNet18, HierarchicalEfficientNetB0))


def _build_param_groups(model, head_lr, backbone_lr_mult):
    """Build optimizer parameter groups with discriminative LR.

    Backbone parameters get head_lr * backbone_lr_mult; everything else
    (trunk + heads, or fc for naive) gets the full head_lr. Frozen
    parameters are still listed; AdamW skips them due to requires_grad.
    """
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if n.startswith('backbone'):
            backbone_params.append(p)
        else:
            head_params.append(p)

    return [
        {'params': backbone_params, 'lr': head_lr * backbone_lr_mult, 'name': 'backbone'},
        {'params': head_params,     'lr': head_lr,                    'name': 'head'},
    ]


def _build_scheduler(optimizer, config):
    """Pick the LR scheduler based on config['scheduler'].

    cosine: standard CosineAnnealingLR over all epochs.
    cosine_warm_restarts: restarts every config['restart_period'] epochs.
    """
    if config['scheduler'] == 'cosine_warm_restarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config['restart_period'], T_mult=1, eta_min=0.0,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])


# Lightweight EMA — equivalent to timm.utils.ModelEmaV2 but no extra
# dependency (since we're already pulling timm for EfficientNet, this is
# defensive in case timm version differs across environments).
class _ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        # Also copy buffers (BN running stats) — important for eval-mode BN
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)


def train_one_fold(model, fold, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_wandb(
        config  = config,
        project = 'isic-2024',
        name    = f"{config['model_name']}-{config['phase']}-fold-{fold}"
    )

    use_libauc = config.get('use_libauc', False)
    use_smoteenn = config.get('use_smoteenn', False)

    # libauc requires per-sample indices through the loss; only enable
    # index-returning dataset when libauc is active.
    return_index = use_libauc

    train_dataset, val_dataset, class_weights, class_names, sample_weights = get_datasets(
        csv_path        = config['csv_path'],
        image_dir       = config['image_dir'],
        fold            = fold,
        n_folds         = config['n_folds'],
        train_transform = train_transform,
        val_transform   = val_transform,
        device          = device.type,
        return_index    = return_index,
    )

    benign_idx = _resolve_benign_idx(class_names)

    # --- DataLoader: sampler toggle ---
    if use_libauc:
        try:
            from libauc.sampler import DualSampler
        except ImportError as exc:
            raise ImportError(
                "use_libauc=True requires libauc sampler support. Install with `pip install libauc`."
            ) from exc

        train_labels = train_dataset.df['iddx_processed'].to_numpy()
        binary_train_labels = (train_labels != benign_idx).astype(np.int64)
        n_pos = int(binary_train_labels.sum())
        n_neg = int(binary_train_labels.shape[0] - n_pos)
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"Fold {fold} has invalid binary class mix for LibAUC: n_pos={n_pos}, n_neg={n_neg}."
            )

        libauc_num_pos = config.get('libauc_num_pos')
        if libauc_num_pos is not None and int(libauc_num_pos) > 0:
            dual_sampler = DualSampler(
                train_dataset,
                batch_size=config['batch_size'],
                labels=binary_train_labels,
                shuffle=True,
                num_pos=int(libauc_num_pos),
            )
        else:
            dual_sampler = DualSampler(
                train_dataset,
                batch_size=config['batch_size'],
                labels=binary_train_labels,
                shuffle=True,
                sampling_rate=float(config.get('libauc_sampling_rate', 0.1)),
            )

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  sampler=dual_sampler, shuffle=False,
                                  num_workers=1, pin_memory=True)
    elif config.get('use_sampler', False):
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  sampler=sampler, shuffle=False,
                                  num_workers=1, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=1, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=1, pin_memory=True)

    model = model.to(device)
    is_hierarchical = _is_hierarchical(model)

    if use_smoteenn and use_libauc:
        raise ValueError("SMOTEENN warmup cannot be combined with LibAUC mode.")

    if use_smoteenn:
        if not hasattr(model, 'extract_features'):
            raise ValueError(
                "SMOTEENN warmup requires model.extract_features(...)."
            )

        print("Extracting embeddings for SMOTEENN...")
        X_res, y_res = apply_smoteenn(
            model,
            train_dataset,
            device=device,
            batch_size=config['batch_size'],
        )

        X_tensor = torch.tensor(X_res, dtype=torch.float32)
        y_tensor = torch.tensor(y_res, dtype=torch.long)
        resampled_loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
        )

        warmup_epochs = int(config.get('smote_warmup_epochs', 5))
        warmup_lr = float(config.get('smote_head_lr', 1e-3))

        if is_hierarchical:
            if not hasattr(model, 'binary_head') or not hasattr(model, 'type_head'):
                raise ValueError(
                    "Hierarchical SMOTEENN warmup requires binary_head and type_head modules."
                )

            head_optimizer = torch.optim.AdamW(
                list(model.binary_head.parameters()) + list(model.type_head.parameters()),
                lr=warmup_lr,
            )
            criterion_head = HierarchicalLoss(
                alpha=config.get('hier_alpha', 1.0),
                beta=config.get('hier_beta', 1.0),
                benign_idx=benign_idx,
                pos_weight=config.get('hier_pos_weight', 100.0),
                type_weights=None,
            ).to(device)

            print("Pre-training hierarchical heads on SMOTEENN-resampled embeddings...")
            for warmup_epoch in range(warmup_epochs):
                model.train()
                for X_batch, y_batch in resampled_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    head_optimizer.zero_grad()
                    warmup_outputs = {
                        'binary_logit': model.binary_head(X_batch).squeeze(-1),
                        'type_logits': model.type_head(X_batch),
                    }
                    loss, _ = criterion_head(warmup_outputs, y_batch.long())
                    loss.backward()
                    head_optimizer.step()

                print(f"  SMOTE head warmup epoch {warmup_epoch + 1}/{warmup_epochs} done")
        else:
            if not hasattr(model, 'fc'):
                raise ValueError(
                    "Naive SMOTEENN warmup requires a model.fc classification head."
                )

            head_optimizer = torch.optim.AdamW(
                model.fc.parameters(),
                lr=warmup_lr,
            )
            criterion_head = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

            print("Pre-training classifier head on SMOTEENN-resampled embeddings...")
            for warmup_epoch in range(warmup_epochs):
                model.train()
                for X_batch, y_batch in resampled_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    head_optimizer.zero_grad()
                    out = model.fc(X_batch)
                    loss = criterion_head(out, y_batch.long())
                    loss.backward()
                    head_optimizer.step()

                print(f"  SMOTE head warmup epoch {warmup_epoch + 1}/{warmup_epochs} done")

    # --- handle initial freeze (before optimizer is built) ---
    if is_hierarchical and config.get('hier_freeze_backbone_epochs', 0) > 0:
        model.freeze_backbone()
        print(f"  [hier] Backbone frozen for first {config['hier_freeze_backbone_epochs']} epochs.")

    # --- optimizer + scheduler ---
    # libauc mode: SOPAs replaces AdamW. SOPAs accepts param groups
    # the same way as torch optimizers, so we share the param-group builder.
    param_groups = _build_param_groups(
        model, head_lr=config['lr'],
        backbone_lr_mult=config.get('backbone_lr_mult', 1.0),
    )

    if use_libauc:
        try:
            from libauc.optimizers import SOPAs
        except ImportError as exc:
            raise ImportError(
                "use_libauc=True requires libauc. Install with `pip install libauc`."
            ) from exc
        optimizer = SOPAs(param_groups, mode='adam', lr=config['lr'],
                          weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=config['lr'],
                                       weight_decay=config['weight_decay'])

    scheduler = _build_scheduler(optimizer, config)

    # --- loss selection ---
    if use_libauc:
        from .pauc_loss_wrapper import PAUCLossWrapper
        criterion = PAUCLossWrapper(
            data_len=len(train_dataset),
            benign_idx=benign_idx,
            type_loss_weight=config.get('hier_type_loss_weight', 0.1),
            is_hierarchical=is_hierarchical,
        ).to(device)
    elif is_hierarchical:
        criterion = HierarchicalLoss(
            alpha      = config['hier_alpha'],
            beta       = config['hier_beta'],
            benign_idx = benign_idx,
            pos_weight = config.get('hier_pos_weight', 100.0),
            type_weights = None,
        ).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # --- EMA setup ---
    use_ema = config.get('use_ema', False)
    ema = _ModelEMA(model, decay=config['ema_decay']) if use_ema else None

    os.makedirs("checkpoints", exist_ok=True)
    best_auc  = 0.0
    best_pauc = 0.0

    for epoch in range(config['epochs']):
        # --- handle unfreeze schedule ---
        # When freeze epoch is reached, either fully unfreeze or partial
        # unfreeze (only late stages). Either way, rebuild the optimizer
        # so newly-trainable params are picked up under the right LR group.
        if is_hierarchical and epoch == config.get('hier_freeze_backbone_epochs', 0) \
                and config.get('hier_freeze_backbone_epochs', 0) > 0:
            if config.get('hier_partial_unfreeze', False):
                model.partial_unfreeze_backbone()
                print(f"  [hier] Backbone partially unfrozen at epoch {epoch + 1} "
                      f"(only late stages trainable).")
            else:
                model.unfreeze_backbone()
                print(f"  [hier] Backbone fully unfrozen at epoch {epoch + 1}.")

            # Rebuild optimizer with same param groups so newly-unfrozen
            # backbone params get the discriminative LR.
            param_groups = _build_param_groups(
                model, head_lr=config['lr'],
                backbone_lr_mult=config.get('backbone_lr_mult', 1.0),
            )
            if use_libauc:
                from libauc.optimizers import SOPAs
                optimizer = SOPAs(param_groups, mode='adam', lr=config['lr'],
                                  weight_decay=config['weight_decay'])
            else:
                optimizer = torch.optim.AdamW(param_groups, lr=config['lr'],
                                               weight_decay=config['weight_decay'])
            scheduler = _build_scheduler(optimizer, config)

        # --- train ---
        model.train()
        train_loss = 0.0
        train_probs = []
        train_preds = []
        train_labels = []
        train_grad_stats = []
        epoch_L_binary = []
        epoch_L_type   = []
        epoch_L_pauc   = []

        for batch in train_loader:
            # Unpack batch — index is present iff return_index was True
            if return_index:
                imgs, labels, indices = batch
                imgs    = imgs.to(device)
                labels  = labels.to(device)
                indices = indices.to(device)
            else:
                imgs, labels = batch
                imgs   = imgs.to(device)
                labels = labels.to(device)
                indices = None

            optimizer.zero_grad()
            outputs = model(imgs)

            # Compute loss + extract probs for metrics
            if use_libauc:
                loss, components = criterion(outputs, labels.long(), indices)
                epoch_L_pauc.append(components['L_pauc'])
                epoch_L_type.append(components['L_type'])
                if is_hierarchical:
                    probs_batch = outputs['probs_5class'].detach()
                else:
                    probs_batch = torch.softmax(outputs.detach(), dim=1)
            elif is_hierarchical:
                loss, components = criterion(outputs, labels.long())
                epoch_L_binary.append(components['L_binary'])
                epoch_L_type.append(components['L_type'])
                probs_batch = outputs['probs_5class'].detach()
            else:
                loss = criterion(outputs, labels.long())
                probs_batch = torch.softmax(outputs.detach(), dim=1)

            loss.backward()
            grad_stats = collect_gradient_stats(model)
            if grad_stats:
                train_grad_stats.append(grad_stats)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            train_loss += loss.item()
            train_probs.extend(probs_batch.cpu().numpy())
            train_preds.extend(probs_batch.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # --- val ---
        # Use EMA weights for validation when EMA is active.
        eval_model = ema.module if ema is not None else model
        eval_model.eval()

        val_loss   = 0.0
        all_probs  = []
        all_preds  = []
        all_labels = []
        all_binary_probs = []  # hierarchical only — raw P(malignant)

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = eval_model(imgs)

                # Validation loss is informational only; we use the
                # ordinary hierarchical/CE loss even in libauc mode so
                # val_loss is comparable across runs.
                if is_hierarchical:
                    if use_libauc:
                        # Approximate val loss with BCE for tracking only
                        is_mal = (labels != benign_idx).float()
                        vloss = nn.functional.binary_cross_entropy_with_logits(
                            outputs['binary_logit'], is_mal,
                        )
                    else:
                        vloss, _ = criterion(outputs, labels.long())
                    probs_batch = outputs['probs_5class']
                    all_binary_probs.extend(torch.sigmoid(outputs['binary_logit']).cpu().numpy())
                else:
                    if use_libauc:
                        vloss = nn.functional.cross_entropy(outputs, labels.long())
                    else:
                        vloss = criterion(outputs, labels.long())
                    probs_batch = torch.softmax(outputs, dim=1)

                val_loss += vloss.item()
                all_probs.extend(probs_batch.cpu().numpy())
                all_preds.extend(probs_batch.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        metrics = {
            'epoch'      : epoch + 1,
            'train/loss' : avg_train_loss,
            'val/loss'   : avg_val_loss,
            'lr/head'    : optimizer.param_groups[-1]['lr'],
            'lr/backbone': optimizer.param_groups[0]['lr'],
        }
        metrics.update(compute_classification_metrics(
            labels=train_labels, preds=train_preds, probs=train_probs,
            class_names=class_names, split='train',
            benign_class_name='benign', pauc_min_tpr=0.8,
        ))
        metrics.update(compute_classification_metrics(
            labels=all_labels, preds=all_preds, probs=all_probs,
            class_names=class_names, split='val',
            benign_class_name='benign', pauc_min_tpr=0.8,
        ))
        metrics.update(summarize_gradient_stats(train_grad_stats, split='train'))

        # Hierarchical-specific extras — split losses and binary-head pAUC
        if is_hierarchical:
            if epoch_L_binary:
                metrics['train/L_binary'] = float(np.mean(epoch_L_binary))
            if epoch_L_type:
                metrics['train/L_type'] = float(np.mean(epoch_L_type))
            if epoch_L_pauc:
                metrics['train/L_pauc'] = float(np.mean(epoch_L_pauc))

            from .logger import _binary_partial_auc_above_tpr
            y_true_mal = (np.asarray(all_labels) != benign_idx).astype(np.int32)
            if np.unique(y_true_mal).size >= 2:
                pauc_raw = _binary_partial_auc_above_tpr(
                    y_true_binary=y_true_mal,
                    y_score=np.asarray(all_binary_probs),
                    min_tpr=0.8,
                )
                if pauc_raw is not None:
                    metrics['val/binary_head/pauc_tpr_ge_80'] = float(pauc_raw)

        # --- W&B diagnostic plots ---
        plots_every_n = config.get('plots_every_n_epochs', 1)
        calibration_every_n = config.get('calibration_every_n_epochs', 5)
        is_last_epoch = (epoch + 1 == config['epochs'])

        if (epoch % plots_every_n == 0) or is_last_epoch:
            include_calibration = (epoch % calibration_every_n == 0) or is_last_epoch
            val_plots = compute_classification_plots(
                labels=all_labels, preds=all_preds, probs=all_probs,
                class_names=class_names, split='val',
                benign_class_name='benign',
                include_calibration=include_calibration,
            )
            metrics.update(val_plots)

        log_metrics(metrics, step=epoch)

        val_auc  = metrics.get('val/auc_macro', float('nan'))
        val_acc  = metrics.get('val/accuracy', float('nan'))
        val_pauc = metrics.get('val/malignant_vs_benign/pauc_tpr_ge_80')
        pauc_text = f" val_pauc80={val_pauc:.4f}" if val_pauc is not None else ""
        if is_hierarchical and 'val/binary_head/pauc_tpr_ge_80' in metrics:
            pauc_text += f" (binary={metrics['val/binary_head/pauc_tpr_ge_80']:.4f})"

        print(f"[{config['model_name']} | fold {fold} | epoch {epoch+1}] "
              f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_auc={val_auc:.4f}{pauc_text}")

        if 'val/auc_macro' in metrics and metrics['val/auc_macro'] > best_auc:
            best_auc = metrics['val/auc_macro']

        if val_pauc is not None and val_pauc > best_pauc:
            best_pauc = val_pauc
            # Save EMA weights when EMA is on, else live model weights.
            checkpoint_state = (ema.module if ema is not None else model).state_dict()
            torch.save(checkpoint_state,
                       f"checkpoints/{config['model_name']}_{config['phase']}_fold{fold}_best.pt")

        scheduler.step()

    finish_wandb()
    print(f"  -> Best pAUC (TPR>=80%): {best_pauc:.4f}  |  Best AUC macro: {best_auc:.4f}")
    return best_pauc
