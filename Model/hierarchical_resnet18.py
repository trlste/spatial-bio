# Model/hierarchical_resnet18.py

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class HierarchicalResNet18(nn.Module):
    """ResNet18 backbone with a dual-head hierarchical classifier.

    The binary head produces P(malignant) as a single sigmoid output,
    which is the clean signal used to compute pAUC. The type head
    produces a 4-way softmax over the malignant classes (BCC,
    Dysplastic, Melanoma, SCC) conditional on the lesion being
    malignant. The two heads share a common trunk so that features
    useful for the cancer-vs-benign boundary also inform the type
    discrimination.

    forward() returns a dict so the training loop can apply the
    hierarchical loss to the raw logits and still pull a composed
    5-class probability vector for the existing metric pipeline:

        P(benign)     = 1 - P(malignant)
        P(cancer_i)   = P(malignant) * P(type_i | malignant)

    The 4 type-head slots map to the 4 malignant classes in their
    original alphabetical order, skipping benign. With the default
    cat.codes ordering (BCC=0, Benign=1, Dysplastic=2, Melanoma=3,
    SCC=4) and benign_idx=1, the map is:
        type_idx 0 -> BCC        (original 0)
        type_idx 1 -> Dysplastic (original 2)
        type_idx 2 -> Melanoma   (original 3)
        type_idx 3 -> SCC        (original 4)

    The trunk can be 1- or 2-layer:
        trunk_dim_2 == 0: 512 -> trunk_dim -> heads
        trunk_dim_2 >  0: 512 -> trunk_dim -> trunk_dim_2 -> heads
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
            raise ValueError("HierarchicalResNet18 requires n_classes >= 2.")
        if not 0 <= benign_idx < n_classes:
            raise ValueError(f"benign_idx {benign_idx} out of range for {n_classes} classes.")

        self.n_classes  = n_classes
        self.benign_idx = benign_idx
        self.n_types    = n_classes - 1  # number of malignant sub-classes

        # Backbone — same pattern as NaiveResNet18, keep everything up to
        # the avgpool so we get a (B, 512, 1, 1) tensor. The Sequential
        # children, in order, are:
        #   [0] conv1  [1] bn1     [2] relu    [3] maxpool
        #   [4] layer1 [5] layer2  [6] layer3  [7] layer4  [8] avgpool
        weights  = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Shared trunk — optionally 2 layers deep.
        trunk_layers = [
            nn.Linear(512, trunk_dim),
            nn.BatchNorm1d(trunk_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]
        trunk_out = trunk_dim
        if trunk_dim_2 and trunk_dim_2 > 0:
            trunk_layers.extend([
                nn.Linear(trunk_dim, trunk_dim_2),
                nn.BatchNorm1d(trunk_dim_2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            trunk_out = trunk_dim_2
        self.trunk = nn.Sequential(*trunk_layers)

        # Binary head — single logit for malignant vs benign.
        self.binary_head = nn.Linear(trunk_out, 1)

        # Type head — logits over the n_types malignant classes.
        self.type_head = nn.Linear(trunk_out, self.n_types)

        # Precompute the list of malignant class indices in the original
        # 5-class label space so forward() can scatter type probs into
        # the right columns without branching.
        self._malignant_cols = [c for c in range(n_classes) if c != benign_idx]

    # --- freeze controls --------------------------------------------------

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def partial_unfreeze_backbone(self):
        """Freeze early stages, unfreeze layer3/layer4/avgpool.

        Children indices in self.backbone:
            0: conv1, 1: bn1, 2: relu, 3: maxpool, 4: layer1, 5: layer2
            6: layer3, 7: layer4, 8: avgpool

        Freezes children 0-5 (everything through layer2), unfreezes 6-8.
        Rationale: early conv stages encode generic edges/textures that
        transfer well from ImageNet; later semantic layers benefit from
        domain-specific fine-tuning.
        """
        for i, child in enumerate(self.backbone):
            requires = i >= 6
            for p in child.parameters():
                p.requires_grad = requires

    def extract_features(self, x):
        """Return shared trunk embeddings before hierarchical heads."""
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return self.trunk(features)

    def forward(self, x):
        features = self.backbone(x)           # (B, 512, 1, 1)
        features = torch.flatten(features, 1) # (B, 512)
        trunk    = self.trunk(features)       # (B, trunk_out)

        binary_logit = self.binary_head(trunk).squeeze(-1)  # (B,)
        type_logits  = self.type_head(trunk)                # (B, n_types)

        # Composed 5-class probabilities — only computed during forward,
        # so the training loop can reuse the existing metric code which
        # expects a softmax-style probability matrix.
        p_malignant = torch.sigmoid(binary_logit)                   # (B,)
        p_type      = torch.softmax(type_logits, dim=-1)            # (B, n_types)

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
