# src/train_one_fold.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.logger import init_wandb, log_metrics, finish_wandb
from src.dataset import get_datasets
from src.transforms import train_transform, val_transform
from sklearn.metrics import accuracy_score, roc_auc_score

def train_one_fold(model, fold, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_wandb(
        config  = config,
        project = 'isic-2024',
        name    = f"{config['model_name']}-fold-{fold}"
    )

    train_dataset, val_dataset, class_weights = get_datasets(
        csv_path        = config['csv_path'],
        image_dir       = config['image_dir'],
        fold            = fold,
        n_folds         = config['n_folds'],
        train_transform = train_transform,
        val_transform   = val_transform,
        device          = device
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)

    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    os.makedirs("checkpoints", exist_ok=True)
    best_auc = 0

    for epoch in range(config['epochs']):
        # --- train ---
        model.train()
        train_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss    = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # --- val ---
        model.eval()
        val_loss   = 0
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
        accuracy = accuracy_score(all_labels, all_preds)        
        auc      = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

        log_metrics({
            'epoch'      : epoch + 1,
            'train/loss' : avg_train_loss,
            'val/loss'   : avg_val_loss,
            'val/accuracy' : accuracy,
            'val/auc_macro': auc,
            'lr'         : optimizer.param_groups[0]['lr'],
        }, step=epoch)

        print(f"[{config['model_name']} | fold {fold} | epoch {epoch+1}] "
              f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
              f"val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(),
                       f"checkpoints/{config['model_name']}_fold{fold}_best.pt")

        scheduler.step()

    finish_wandb()
    return best_auc