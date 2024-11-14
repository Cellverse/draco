import torch
import torch.nn as nn

__all__ = [
    "LayerNorm2d",
]


class LayerNorm2d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).square().mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
