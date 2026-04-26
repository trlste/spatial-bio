# src/train_one_fold.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .logger import (
    collect_gradient_stats,
    compute_classification_metrics,
    finish_wandb,
    init_wandb,
    log_metrics,
    summarize_gradient_stats,
)
from .dataset import get_datasets
from .transforms import train_transform, val_transform
from .focal_loss import FocalLoss
from .dataset import get_datasets, apply_smoteenn, apply_smoteenn_naive
from torch.utils.data import TensorDataset

def train_one_fold(model, fold, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    init_wandb(
        config  = config,
        project = 'isic-2024',
        name    = f"{config['model_name']}-fold-{fold}"
    )

    train_dataset, val_dataset, class_weights, class_names = get_datasets(
        csv_path        = config['csv_path'],
        image_dir       = config['image_dir'],
        fold            = fold,
        n_folds         = config['n_folds'],
        train_transform = train_transform,
        val_transform   = val_transform,
        device          = device.type
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = model.to(device)

    print("Extracting embeddings for SMOTEENN...")
    X_res, y_res = apply_smoteenn(model, train_dataset, device=device,
                                  batch_size=config['batch_size'])

    # Wrap resampled embeddings as a TensorDataset for the head trainer
    X_tensor = torch.tensor(X_res, dtype=torch.float32)
    y_tensor = torch.tensor(y_res, dtype=torch.long)
    resampled_dataset = TensorDataset(X_tensor, y_tensor)
    resampled_loader  = DataLoader(resampled_dataset,
                                   batch_size=config['batch_size'],
                                   shuffle=True, num_workers=0)
    
    print("Pre-training fc head on SMOTEENN-resampled embeddings...")
    head_optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
    criterion_head = FocalLoss(gamma=2.0, alpha=class_weights)

    for epoch in range(5):                               # short warmup, ~5 epochs
        model.train()
        for X_batch, y_batch in resampled_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            head_optimizer.zero_grad()
            out  = model.fc(X_batch)                     # skip backbone entirely
            loss = criterion_head(out, y_batch)
            loss.backward()
            head_optimizer.step()
        print(f"  Head warmup epoch {epoch+1}/5 done")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)

    os.makedirs("checkpoints", exist_ok=True)
    best_auc = 0

    for epoch in range(config['epochs']):
        # --- train ---
        model.train()
        train_loss = 0.0
        train_probs = []
        train_preds = []
        train_labels = []
        train_grad_stats = []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss    = criterion(outputs, labels.long())
            loss.backward()
            grad_stats = collect_gradient_stats(model)
            if grad_stats:
                train_grad_stats.append(grad_stats)
            optimizer.step()

            train_loss += loss.item()
            train_probs.extend(torch.softmax(outputs.detach(), dim=1).cpu().numpy())
            train_preds.extend(outputs.detach().argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # --- val ---
        model.eval()
        val_loss   = 0.0
        all_probs  = []
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                loss    = criterion(outputs, labels.long())

                val_loss += loss.item()
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())  # softmax not sigmoid, full prob distribution
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())          # predicted class index
                all_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        metrics = {
            'epoch'      : epoch + 1,
            'train/loss' : avg_train_loss,
            'val/loss'   : avg_val_loss,
            'lr'         : optimizer.param_groups[0]['lr'],
        }
        metrics.update(compute_classification_metrics(
            labels=train_labels,
            preds=train_preds,
            probs=train_probs,
            class_names=class_names,
            split='train',
            benign_class_name='benign',
            pauc_min_tpr=0.8,
        ))
        metrics.update(compute_classification_metrics(
            labels=all_labels,
            preds=all_preds,
            probs=all_probs,
            class_names=class_names,
            split='val',
            benign_class_name='benign',
            pauc_min_tpr=0.8,
        ))
        metrics.update(summarize_gradient_stats(train_grad_stats, split='train'))

        log_metrics(metrics, step=epoch)

        val_auc = metrics.get('val/auc_macro', float('nan'))
        val_acc = metrics.get('val/accuracy', float('nan'))
        val_pauc = metrics.get('val/malignant_vs_benign/pauc_tpr_ge_80')
        pauc_text = f" val_pauc80={val_pauc:.4f}" if val_pauc is not None else ""

        print(f"[{config['model_name']} | fold {fold} | epoch {epoch+1}] "
              f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_auc={val_auc:.4f}{pauc_text}")

        if 'val/auc_macro' in metrics and metrics['val/auc_macro'] > best_auc:
            best_auc = metrics['val/auc_macro']
            torch.save(model.state_dict(),
                       f"checkpoints/{config['model_name']}_fold{fold}_best.pt")

        scheduler.step()

    finish_wandb()
    return best_auc