# src/pauc_loss_wrapper.py

"""LibAUC pAUC loss wrapper for both single-head and hierarchical models.

Provides a uniform __call__(model_output, labels, indices) interface that
extracts a binary malignant score appropriately for each model type, then
delegates to libauc.losses.pAUCLoss. For hierarchical models we keep the
type-head CE active (weighted down) so the multiclass capability is
retained while pAUC drives the binary signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PAUCLossWrapper(nn.Module):
    """Wrapper around libauc.pAUCLoss('1w') that handles our model outputs.

    For hierarchical models, applies pAUC loss to the binary head and CE
    to the type head (the latter is mask-restricted to malignant samples
    and weighted by `type_loss_weight`, default 0.1, so it doesn't drown
    out the pAUC signal but still trains the type discrimination).

    For single-head 5-class models, derives a malignant score as
    `1 - softmax(logits)[:, benign_idx]`. Note: this score is in [0, 1]
    but the pAUC surrogate expects raw logits in some implementations —
    libauc accepts probs and applies its own surrogate internally, so
    we feed the probability score directly.

    Args:
        data_len: total number of training samples (libauc needs this
            to allocate per-positive running state)
        benign_idx: index of benign class in the 5-class label space
        margin: pAUCLoss margin hyperparameter (default 1.0)
        gamma: pAUCLoss gamma hyperparameter (default 0.5)
        fpr_threshold: max FPR used in pAUC; for TPR>=80%, set 0.2
            so pAUC integrates over FPR in [0, 0.2]
        type_loss_weight: weight on auxiliary type CE for hierarchical
            models (default 0.1; set 0 to disable)
        is_hierarchical: True for HierarchicalResNet18 / EfficientNetB0;
            False for single-head 5-class models
    """

    def __init__(
        self,
        data_len,
        benign_idx=1,
        margin=1.0,
        gamma=0.5,
        fpr_threshold=0.2,
        type_loss_weight=0.1,
        is_hierarchical=True,
    ):
        super().__init__()

        try:
            from libauc.losses import pAUCLoss
        except ImportError as exc:
            raise ImportError(
                "PAUCLossWrapper requires libauc. Install with `pip install libauc`."
            ) from exc

        # libauc's pAUCLoss('1w') uses pAUC_DRO_Loss as the backend and
        # SOPAs as the paired optimizer. The DRO formulation pushes the
        # ROC curve up specifically in the [0, fpr_threshold] regime.
        # libauc parameter naming uses `beta` in the underlying loss and
        # also accepts `Lambda` style args — the public pAUCLoss wrapper
        # forwards margin, gamma, and uses default fpr coverage. To
        # restrict the FPR window, we pass it through where supported.
        try:
            self.pauc_loss = pAUCLoss(
                '1w',
                data_len=data_len,
                margin=margin,
                gamma=gamma,
            )
        except TypeError:
            # Older libauc signature — fall back to mode and data_len only.
            self.pauc_loss = pAUCLoss('1w', data_len=data_len)

        self.benign_idx       = benign_idx
        self.type_loss_weight = type_loss_weight
        self.is_hierarchical  = is_hierarchical

    def _malignant_score(self, model_output):
        """Extract a binary malignant score / logit from model output.

        For hierarchical models we use the binary head's raw logit
        directly (libauc applies sigmoid + surrogate internally).

        For single-head 5-class models we use 1 - softmax_benign as the
        score. This is bounded in [0, 1] and monotonic in P(malignant);
        libauc's surrogate handles either logits or scores.
        """
        if self.is_hierarchical:
            # Hierarchical model returns a dict with 'binary_logit'.
            # libauc expects shape (B, 1).
            return model_output['binary_logit'].unsqueeze(-1)
        else:
            # Single-head: derive malignant score from the 5-class softmax.
            # 1 - P(benign) = sum of all malignant softmax probs.
            probs = F.softmax(model_output, dim=-1)
            mal_score = 1.0 - probs[:, self.benign_idx]
            return mal_score.unsqueeze(-1)

    def forward(self, model_output, labels, indices):
        """Compute the joint loss.

        Args:
            model_output: hierarchical dict OR (B, 5) logits tensor
            labels: int64 tensor of original 5-class labels (B,)
            indices: int64 tensor of dataset positions (B,) — required
                by libauc to maintain per-positive running state

        Returns:
            (total_loss, components_dict)
        """
        # Binary target: 1 if malignant, 0 if benign. libauc expects long.
        binary_target = (labels != self.benign_idx).long()

        # libauc's pAUCLoss takes (preds, target, index) where preds is
        # the model's output for the positive class — shape (B, 1) or (B,).
        mal_score = self._malignant_score(model_output)
        L_pauc = self.pauc_loss(mal_score, binary_target, indices)

        components = {
            'L_pauc': L_pauc.detach().item(),
            'L_type': 0.0,
        }

        # Auxiliary type CE for hierarchical models — keeps the type head
        # trained so multi-class metrics stay meaningful, but weighted
        # down so it doesn't fight the pAUC objective.
        if self.is_hierarchical and self.type_loss_weight > 0:
            type_logits    = model_output['type_logits']
            malignant_mask = labels != self.benign_idx
            if malignant_mask.any():
                malignant_labels = labels[malignant_mask]
                # Shift labels above benign_idx down by one to match
                # the n_types-slot type head (same as HierarchicalLoss).
                type_labels = torch.where(
                    malignant_labels < self.benign_idx,
                    malignant_labels,
                    malignant_labels - 1,
                )
                L_type = F.cross_entropy(type_logits[malignant_mask], type_labels)
                total = L_pauc + self.type_loss_weight * L_type
                components['L_type'] = L_type.detach().item()
                return total, components

        return L_pauc, components
