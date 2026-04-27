# src/logger.py
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

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
            print(f"[wandb] WANDB_API_KEY login failed ({exc}). Falling back to existing W&B credentials.")
    else:
        print('[wandb] WANDB_API_KEY not set. Trying existing W&B login credentials.')

    try:
        wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"[wandb] init failed ({exc}). Running with W&B disabled.")
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

def _calibration_reliability_figure(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    title: str = 'Reliability diagram — P(malignant)',
):
    """Build a matplotlib figure for a calibration reliability diagram.

    Bins predicted probabilities into n_bins equal-width intervals,
    computes the empirical positive rate per bin, and plots against the
    mean predicted probability per bin. A well-calibrated model tracks
    the y=x diagonal; above the diagonal means under-confident, below
    means over-confident.

    Returns None when matplotlib is unavailable or the data is
    degenerate (single-class or empty).
    """
    if y_true_binary.size == 0 or np.unique(y_true_binary).size < 2:
        return None

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids   = np.clip(np.digitize(y_score, bin_edges[1:-1]), 0, n_bins - 1)

    mean_pred   = np.full(n_bins, np.nan)
    empirical   = np.full(n_bins, np.nan)
    bin_counts  = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        bin_counts[b] = count
        if count > 0:
            mean_pred[b] = float(y_score[mask].mean())
            empirical[b] = float(y_true_binary[mask].mean())

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(5.2, 5.5), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )

    valid = ~np.isnan(mean_pred)
    ax_main.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    ax_main.plot(mean_pred[valid], empirical[valid], marker='o', label='Model')
    ax_main.set_ylabel('Empirical positive rate')
    ax_main.set_ylim(-0.02, 1.02)
    ax_main.set_title(title)
    ax_main.legend(loc='upper left')
    ax_main.grid(True, alpha=0.3)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax_hist.bar(bin_centers, bin_counts, width=1.0 / n_bins, edgecolor='black', alpha=0.6)
    ax_hist.set_xlabel('Predicted P(malignant)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_xlim(-0.02, 1.02)
    ax_hist.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def compute_classification_plots(
    labels: Sequence[int],
    preds: Sequence[int],
    probs: Sequence[Sequence[float]],
    class_names: Sequence[str],
    split: str,
    benign_class_name: str = 'benign',
    include_calibration: bool = False,
) -> Dict[str, object]:
    """Build W&B plot objects for the current validation state.

    Produces a confusion matrix (via wandb.plot), ROC curves, PR curves
    (both multiclass one-vs-rest and a dedicated malignant-vs-benign
    version), and — when include_calibration=True — a reliability
    diagram for P(malignant). Returned as a dict ready to be merged
    into the metrics dict and passed to wandb.log().

    The dict may be empty if wandb is running in disabled mode and
    cannot accept plot objects; callers should not rely on specific
    keys being present.
    """
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(preds, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)

    plots: Dict[str, object] = {}
    if y_true.size == 0:
        return plots

    # --- Confusion matrix over the full 5-class label space ---
    sanitized_names = [_sanitize_class_name(n) for n in class_names]
    try:
        plots[f'{split}/plots/confusion_matrix'] = wandb.plot.confusion_matrix(
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=sanitized_names,
        )
    except Exception:
        # W&B can be disabled or offline — plots are best-effort.
        pass

    # --- Multiclass ROC (one-vs-rest) — wandb.plot handles the plumbing ---
    try:
        plots[f'{split}/plots/roc_multiclass'] = wandb.plot.roc_curve(
            y_true=y_true,
            y_probas=y_prob,
            labels=sanitized_names,
        )
    except Exception:
        pass

    try:
        plots[f'{split}/plots/pr_multiclass'] = wandb.plot.pr_curve(
            y_true=y_true,
            y_probas=y_prob,
            labels=sanitized_names,
        )
    except Exception:
        pass

    # --- Malignant-vs-benign binary curves ---
    # This is the clinically meaningful view — matches how pAUC is framed.
    normalized_names = [str(n).strip().lower() for n in class_names]
    benign_key = benign_class_name.strip().lower()
    benign_idx = None
    if benign_key in normalized_names:
        benign_idx = normalized_names.index(benign_key)
    else:
        for idx, name in enumerate(normalized_names):
            if benign_key in name:
                benign_idx = idx
                break

    if benign_idx is not None and benign_idx < y_prob.shape[1]:
        y_true_binary = (y_true != benign_idx).astype(np.int32)
        malignant_score = 1.0 - y_prob[:, benign_idx]

        if np.unique(y_true_binary).size >= 2:
            # ROC curve — plot as a W&B line series so it animates across epochs
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, malignant_score)
                data = [[float(a), float(b)] for a, b in zip(fpr, tpr)]
                table = wandb.Table(data=data, columns=['fpr', 'tpr'])
                plots[f'{split}/plots/roc_malignant_vs_benign'] = wandb.plot.line(
                    table, 'fpr', 'tpr',
                    title='ROC — malignant vs benign',
                )
            except Exception:
                pass

            # PR curve
            try:
                precision, recall, _ = precision_recall_curve(y_true_binary, malignant_score)
                data = [[float(r), float(p)] for r, p in zip(recall, precision)]
                table = wandb.Table(data=data, columns=['recall', 'precision'])
                plots[f'{split}/plots/pr_malignant_vs_benign'] = wandb.plot.line(
                    table, 'recall', 'precision',
                    title='PR — malignant vs benign',
                )
            except Exception:
                pass

            # Calibration — expensive enough that we only emit it when asked.
            if include_calibration:
                fig = _calibration_reliability_figure(y_true_binary, malignant_score)
                if fig is not None:
                    try:
                        plots[f'{split}/plots/calibration_malignant'] = wandb.Image(fig)
                    except Exception:
                        pass
                    finally:
                        try:
                            import matplotlib.pyplot as plt
                            plt.close(fig)
                        except ImportError:
                            pass

    return plots
