import torch.nn.functional as F
import torch
import torch.nn as nn

# https://amaarora.github.io/posts/2020-06-29-FocalLoss.html
class FocalLossMultiClass(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLossMultiClass, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            ce_loss = alpha_t * ce_loss

        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()