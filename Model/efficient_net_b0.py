# Model/efficient_net_b0.py

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False


def _build_efficientnet_backbone(pretrained: bool = True):
    """Create a tf_efficientnet_b0_ns backbone with the classifier stripped.

    timm's `num_classes=0` returns the model with global pooling included
    but no final Linear layer, producing a (B, 1280) feature vector when
    called as `backbone(x)`. The 'ns' suffix selects the noisy-student
    pretrained weights, which are the strongest publicly available B0
    weights and consistently outperform vanilla ImageNet B0 on ISIC.
    """
    if not _HAS_TIMM:
        raise ImportError(
            "EfficientNetB0 requires timm. Install with `pip install timm`."
        )
    return timm.create_model(
        'tf_efficientnet_b0_ns',
        pretrained=pretrained,
        num_classes=0,
    )


def _build_trunk(in_dim: int, trunk_dim: int, trunk_dim_2: int, dropout: float):
    """Build the shared trunk module.

    Args:
        in_dim: backbone feature dim (512 for ResNet18, 1280 for B0)
        trunk_dim: width of first hidden layer
        trunk_dim_2: width of second hidden layer; if 0 or None, single-layer trunk
        dropout: dropout probability applied after each ReLU

    Returns:
        (nn.Sequential trunk module, output dim of trunk)
    """
    layers = [
        nn.Linear(in_dim, trunk_dim),
        nn.BatchNorm1d(trunk_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    ]
    out_dim = trunk_dim
    if trunk_dim_2 and trunk_dim_2 > 0:
        layers.extend([
            nn.Linear(trunk_dim, trunk_dim_2),
            nn.BatchNorm1d(trunk_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ])
        out_dim = trunk_dim_2
    return nn.Sequential(*layers), out_dim


class EfficientNetB0(nn.Module):
    """Single-head EfficientNet-B0 — drop-in replacement for NaiveResNet18.

    Same constructor signature so it slots into the existing else-branch
    in training.py with no special-casing.
    """

    def __init__(self, n_classes=5, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = _build_efficientnet_backbone(pretrained=pretrained)
        # tf_efficientnet_b0_ns feature dim is 1280
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(1280, n_classes)

    def forward(self, x):
        x = self.backbone(x)        # (B, 1280) — pool already applied by timm
        x = self.dropout(x)
        return self.fc(x)

    def extract_features(self, x):
        """Return pooled backbone embeddings before dropout/fc."""
        return self.backbone(x)


class HierarchicalEfficientNetB0(nn.Module):
    """Dual-head EfficientNet-B0 — drop-in for HierarchicalResNet18.

    Same forward() contract: returns dict with binary_logit, type_logits,
    probs_5class. Supports the same freeze/unfreeze/partial_unfreeze API
    so train_one_fold.py can stay agnostic to the backbone.
    """

    def __init__(
        self,
        n_classes=5,
        benign_idx=1,
        dropout=0.4,
        pretrained=True,
        trunk_dim=256,
        trunk_dim_2=0,
    ):
        super().__init__()
        if n_classes < 2:
            raise ValueError("Need n_classes >= 2.")
        if not 0 <= benign_idx < n_classes:
            raise ValueError(f"benign_idx {benign_idx} out of range.")

        self.n_classes  = n_classes
        self.benign_idx = benign_idx
        self.n_types    = n_classes - 1

        self.backbone = _build_efficientnet_backbone(pretrained=pretrained)
        self.trunk, trunk_out = _build_trunk(
            in_dim=1280, trunk_dim=trunk_dim,
            trunk_dim_2=trunk_dim_2, dropout=dropout,
        )

        self.binary_head = nn.Linear(trunk_out, 1)
        self.type_head   = nn.Linear(trunk_out, self.n_types)
        self._malignant_cols = [c for c in range(n_classes) if c != benign_idx]

    # --- freeze controls --------------------------------------------------
    # timm EfficientNet has named submodules: conv_stem, bn1, blocks (a
    # ModuleList of 7 stages), conv_head, bn2, global_pool. Partial unfreeze
    # leaves the last two stages + conv_head + bn2 trainable, freezing the
    # earlier feature-extraction layers.

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def partial_unfreeze_backbone(self):
        # Freeze everything first, then selectively unfreeze the late stages.
        for p in self.backbone.parameters():
            p.requires_grad = False

        # blocks[5] and blocks[6] are the last two stages (highest semantic
        # level), conv_head + bn2 are the post-block projection. These are
        # the layers most likely to need domain adaptation for dermoscopy.
        late_modules = []
        if hasattr(self.backbone, 'blocks'):
            n_stages = len(self.backbone.blocks)
            # Unfreeze the last 2 stages — for B0 this is blocks[5] and blocks[6].
            for i in range(max(0, n_stages - 2), n_stages):
                late_modules.append(self.backbone.blocks[i])
        for name in ('conv_head', 'bn2'):
            if hasattr(self.backbone, name):
                late_modules.append(getattr(self.backbone, name))

        for module in late_modules:
            for p in module.parameters():
                p.requires_grad = True

    def extract_features(self, x):
        """Return shared trunk embeddings before hierarchical heads."""
        features = self.backbone(x)
        return self.trunk(features)

    # --- forward ----------------------------------------------------------

    def forward(self, x):
        features = self.backbone(x)            # (B, 1280)
        trunk    = self.trunk(features)

        binary_logit = self.binary_head(trunk).squeeze(-1)
        type_logits  = self.type_head(trunk)

        p_malignant = torch.sigmoid(binary_logit)
        p_type      = torch.softmax(type_logits, dim=-1)

        probs_5class = torch.empty(
            x.size(0), self.n_classes,
            device=binary_logit.device, dtype=p_malignant.dtype,
        )
        probs_5class[:, self.benign_idx] = 1.0 - p_malignant
        for type_idx, class_idx in enumerate(self._malignant_cols):
            probs_5class[:, class_idx] = p_malignant * p_type[:, type_idx]

        return {
            'binary_logit': binary_logit,
            'type_logits' : type_logits,
            'probs_5class': probs_5class,
        }
