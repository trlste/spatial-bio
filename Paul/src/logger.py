# src/logger.py
import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

def init_wandb(config: dict, project: str = 'isic-2024', name: Optional[str] = None):
    api_key = os.getenv('WANDB_API_KEY')
    init_kwargs = {
        'project': project,
        'name': name,
        'config': config,
    }

    if api_key:
        try:
            wandb.login(key=api_key)
        except Exception as exc:
            print(f"[wandb] login failed ({exc}). Running with W&B disabled.")
            init_kwargs['mode'] = 'disabled'
    else:
        print('[wandb] WANDB_API_KEY not set. Running with W&B disabled.')
        init_kwargs['mode'] = 'disabled'

    wandb.init(**init_kwargs)
    try:
        wandb.define_metric('epoch')
        wandb.define_metric('*', step_metric='epoch')
    except Exception:
        # Keep training resilient in disabled/offline modes.
        pass

def log_metrics(metrics: dict, step: Optional[int] = None):
    wandb.log(metrics, step=step)

def finish_wandb():
    wandb.finish()


def _sanitize_class_name(name: str) -> str:
    sanitized = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(name)).strip('_')
    return sanitized or 'unknown'


def _binary_partial_auc_above_tpr(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    min_tpr: float = 0.8,
) -> Optional[float]:
    if not (0.0 <= min_tpr < 1.0):
        raise ValueError('min_tpr must be in [0, 1).')

    if np.unique(y_true_binary).size < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    normalized_tpr = np.clip((tpr - min_tpr) / (1.0 - min_tpr), 0.0, 1.0)
    return float(np.trapz(normalized_tpr, fpr))


def compute_classification_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    probs: Sequence[Sequence[float]],
    class_names: Sequence[str],
    split: str,
    benign_class_name: str = 'benign',
    pauc_min_tpr: float = 0.8,
) -> Dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(preds, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)

    if y_true.size == 0:
        return {}

    metrics: Dict[str, float] = {
        f'{split}/accuracy': float(accuracy_score(y_true, y_pred)),
    }

    try:
        metrics[f'{split}/auc_macro'] = float(
            roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        )
    except ValueError:
        # Macro AUC is undefined when one or more classes are missing.
        pass

    n_classes = min(y_prob.shape[1], len(class_names))
    for class_idx in range(n_classes):
        class_name = _sanitize_class_name(class_names[class_idx])
        mask = y_true == class_idx
        support = int(mask.sum())

        metrics[f'{split}/class_{class_name}/support'] = float(support)
        if support > 0:
            metrics[f'{split}/class_{class_name}/accuracy'] = float((y_pred[mask] == class_idx).mean())

        y_true_binary = (y_true == class_idx).astype(np.int32)
        if np.unique(y_true_binary).size < 2:
            continue

        try:
            metrics[f'{split}/class_{class_name}/auc'] = float(
                roc_auc_score(y_true_binary, y_prob[:, class_idx])
            )
        except ValueError:
            continue

    normalized_names = [str(name).strip().lower() for name in class_names]
    benign_class_name = benign_class_name.strip().lower()
    benign_idx = None

    if benign_class_name in normalized_names:
        benign_idx = normalized_names.index(benign_class_name)
    else:
        for idx, name in enumerate(normalized_names):
            if benign_class_name in name:
                benign_idx = idx
                break

    if benign_idx is not None and benign_idx < y_prob.shape[1]:
        y_true_malignant = (y_true != benign_idx).astype(np.int32)
        malignant_scores = 1.0 - y_prob[:, benign_idx]

        if np.unique(y_true_malignant).size >= 2:
            try:
                metrics[f'{split}/malignant_vs_benign/auc'] = float(
                    roc_auc_score(y_true_malignant, malignant_scores)
                )
            except ValueError:
                pass

            pauc = _binary_partial_auc_above_tpr(
                y_true_binary=y_true_malignant,
                y_score=malignant_scores,
                min_tpr=pauc_min_tpr,
            )
            if pauc is not None:
                threshold_pct = int(round(100 * pauc_min_tpr))
                metrics[f'{split}/malignant_vs_benign/pauc_tpr_ge_{threshold_pct}'] = float(pauc)

    return metrics


def collect_gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
    grad_l2_sq_sum = 0.0
    grad_abs_sum = 0.0
    grad_count = 0
    grad_nonzero = 0
    grad_max_abs = 0.0

    for param in model.parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach()
        abs_grad = grad.abs()

        grad_l2_sq_sum += grad.pow(2).sum().item()
        grad_abs_sum += abs_grad.sum().item()
        grad_count += grad.numel()
        grad_nonzero += torch.count_nonzero(grad).item()
        grad_max_abs = max(grad_max_abs, abs_grad.max().item())

    if grad_count == 0:
        return {}

    return {
        'grad_global_l2_norm': float(grad_l2_sq_sum ** 0.5),
        'grad_mean_abs': float(grad_abs_sum / grad_count),
        'grad_max_abs': float(grad_max_abs),
        'grad_nonzero_fraction': float(grad_nonzero / grad_count),
    }


def summarize_gradient_stats(gradient_stats_per_batch: Sequence[Dict[str, float]], split: str) -> Dict[str, float]:
    if not gradient_stats_per_batch:
        return {}

    summary: Dict[str, float] = {}
    keys = set()
    for batch_stats in gradient_stats_per_batch:
        keys.update(batch_stats.keys())

    for key in sorted(keys):
        values = [stats[key] for stats in gradient_stats_per_batch if key in stats]
        if not values:
            continue
        summary[f'{split}/{key}_mean'] = float(np.mean(values))
        summary[f'{split}/{key}_max'] = float(np.max(values))

    return summary