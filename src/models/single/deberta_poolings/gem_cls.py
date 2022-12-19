import torch
import torch.nn as nn


class GemClsPooling(nn.Module):
    def __init__(self, dim=1, p=3, eps=1e-6):
        super(GemClsPooling, self).__init__()
        self.dim = dim
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=True)
        self.eps = eps

    def forward(self, last_hidden_state, attention_mask):
        # CLSPool
        cls_return = last_hidden_state[:, 0, :]

        # GeMPool
        attention_mask[:, 0] = 0
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.shape)
        last_hidden_state = (last_hidden_state.clamp(min=self.eps) * attention_mask_expanded).pow(self.p).sum(self.dim)
        gem_return = last_hidden_state / attention_mask_expanded.sum(self.dim).clip(min=self.eps)
        gem_return = gem_return.pow(1 / self.p)
        return torch.cat([cls_return, gem_return], dim=1)
