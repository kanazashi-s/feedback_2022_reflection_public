import torch
import torch.nn as nn


class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta: float, target_weights=[0.21, 0.16, 0.10, 0.16, 0.21, 0.16]):
        """
        https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/smooth_l1_loss.py
        """

        super().__init__()
        self.beta = beta
        self.target_weights = torch.Tensor(target_weights).cuda()

    def forward(self, pred, target):
        if self.beta < 1e-5:
            loss = torch.abs(pred - target)
        else:
            n = torch.abs(pred - target)
            cond = n < self.beta
            loss = torch.where(cond, torch.pow(0.5 * n, 2) / self.beta, n - 0.5 * self.beta)

        loss = torch.sum(loss * self.target_weights)
        return loss
