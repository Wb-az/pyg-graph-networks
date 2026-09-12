"""Custom loss functions not provided efficiently enough by third-party packages.

MultiClassFocalLoss replaces torch_focalloss's implementation: that package's
weighted-mean reduction computes its normalising denominator with a Python
for-loop over every sample in the batch (`Tensor([alpha[val] ... for val in
target])`), which is fine for a small batch but is a severe bottleneck for
full-batch training on a graph with tens of thousands of training nodes, it
was misdiagnosed as MPS-specific slowness before being traced to that loop.
This implementation only supports what this project's training scripts
actually use (class-index targets, optional per-class weights, mean
reduction), which is what keeps it simple enough to get right and fully
vectorised, unlike the general-purpose library it replaces.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiClassFocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017, "Focal Loss for Dense Object
    Detection") for class-index targets.

    :param gamma: focusing parameter; 0 makes this identical to
        (weighted) cross-entropy, larger values down-weight easy examples
        more strongly.
    :param weight: optional per-class weights, shape ``[num_classes]``,
        same convention as ``CrossEntropyLoss(weight=...)``.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """:param logits: unnormalised scores, shape ``[batch, num_classes]``.
        :param target: class indices, shape ``[batch]``.
        """
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        p_t = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        focal = (1 - p_t) ** self.gamma * ce
        if self.weight is None:
            return focal.mean()
        # Same weighted-mean convention as CrossEntropyLoss(weight=...):
        # normalise by the sum of each sample's class weight, not the count,
        # so a class-imbalanced batch doesn't silently down-weight the loss.
        return focal.sum() / self.weight[target].sum()
