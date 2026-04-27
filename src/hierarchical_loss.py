# src/hierarchical_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalLoss(nn.Module):
    """Dual-head loss for HierarchicalResNet18.

    Combines:
      * BCE-with-logits on the binary head (every sample contributes)
      * Cross-entropy on the type head (only malignant samples contribute)

    The original 5-class labels are decomposed internally:
      is_malignant   = (label != benign_idx)             # for BCE target
      type_label     = label with benign_idx removed     # for CE target

    The type-label remap shifts class indices above benign_idx down by
    one so the result matches the n_types-slot type head. With default
    ordering (benign_idx=1), original labels {0, 2, 3, 4} become type
    labels {0, 1, 2, 3}.

    The L_type component is zeroed out if a batch happens to contain no
    malignant samples. In practice the WeightedRandomSampler makes this
    nearly impossible, but it keeps the loss numerically well-defined.
    """

    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        benign_idx=1,
        pos_weight=None,
        type_weights=None,
    ):
        super().__init__()
        self.alpha      = alpha
        self.beta       = beta
        self.benign_idx = benign_idx

        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
        if type_weights is not None:
            type_weights = torch.as_tensor(type_weights, dtype=torch.float32)

        # register_buffer so .to(device) moves these along with the module
        self.register_buffer('pos_weight',   pos_weight,   persistent=False)
        self.register_buffer('type_weights', type_weights, persistent=False)

    def forward(self, outputs, labels):
        """Compute the joint loss.

        Args:
            outputs: dict returned by HierarchicalResNet18.forward with
                keys 'binary_logit' (B,) and 'type_logits' (B, n_types).
            labels: int64 tensor of original 5-class labels, shape (B,).

        Returns:
            total_loss: scalar tensor (with grad)
            components: dict with 'L_binary' and 'L_type' scalar floats
                for logging.
        """
        binary_logit = outputs['binary_logit']
        type_logits  = outputs['type_logits']

        is_malignant = (labels != self.benign_idx).float()

        # --- Binary loss — every sample contributes ---
        L_binary = F.binary_cross_entropy_with_logits(
            binary_logit, is_malignant,
            pos_weight=self.pos_weight,
        )

        # --- Type loss — only malignant samples contribute ---
        malignant_mask = labels != self.benign_idx
        if malignant_mask.any():
            malignant_labels = labels[malignant_mask]
            # Shift indices above benign_idx down by one.
            type_labels = torch.where(
                malignant_labels < self.benign_idx,
                malignant_labels,
                malignant_labels - 1,
            )
            L_type = F.cross_entropy(
                type_logits[malignant_mask],
                type_labels,
                weight=self.type_weights,
            )
        else:
            # Preserve graph — zero loss with grad attached to avoid
            # "no_grad" warnings on rare all-benign batches.
            L_type = type_logits.sum() * 0.0

        total_loss = self.alpha * L_binary + self.beta * L_type
        components = {
            'L_binary': L_binary.detach().item(),
            'L_type'  : L_type.detach().item(),
        }
        return total_loss, components
