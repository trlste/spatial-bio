import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean', 
        ignore_index: int = -100
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # Using register_buffer ensures alpha moves to GPU with the model
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, C, ...) raw logits
        targets: (batch_size, ...) class indices
        """
        # 1. Handle multi-dimensional inputs (e.g., Segmentation maps)
        if inputs.ndim > 2:
            c = inputs.shape[1]
            # Move C to the last dimension and flatten
            inputs = inputs.transpose(1, -1).reshape(-1, c)
            targets = targets.reshape(-1)

        # 2. Log-Softmax is numerically more stable than exp()
        log_p = F.log_softmax(inputs, dim=-1)
        
        # 3. Gather the log probabilities for the actual target classes
        # This is essentially the NLL part of CrossEntropy
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 4. Calculate pt (probability of the true class)
        pt = log_pt.exp()

        # 5. Compute the focal weight: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # 6. Full loss: -alpha * (1 - pt)^gamma * log(pt)
        loss = -focal_term * log_pt

        # 7. Apply class weights (alpha)
        if self.alpha is not None:
            # Pick the weight corresponding to each target class
            batch_alpha = self.alpha[targets]
            loss = loss * batch_alpha

        # 8. Handle ignore_index
        if self.ignore_index != -100:
            mask = (targets != self.ignore_index).float()
            loss = loss * mask
            # We must be careful with the mean if we are ignoring indices
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)

        # 9. Final reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss