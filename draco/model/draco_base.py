from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn


class DenoisingReconstructionAutoencoderVisionTransformerBase(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        return {"cls_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> dict:
        return dict(
            stem=r'^(?:_orig_mod\.)?cls_token|^(?:_orig_mod\.)?pos_embed|^(?:_orig_mod\.)?patch_embed',
            blocks=[(r'^(?:_orig_mod\.)?blocks\.(\d+)', None), (r'^(?:_orig_mod\.)?norm', (99999,))]
        )

    @classmethod
    def random_masking(cls, x: torch.Tensor, mask_ratio: float) -> torch.BoolTensor:
        B, L = x.shape[:2]
        num_masked = int(L * mask_ratio)

        noise = torch.rand(B, L, device=x.device)
        rank = noise.argsort(dim=1)
        mask = rank < num_masked

        return mask

    @abstractmethod
    def forward(self) -> None:
        raise NotImplementedError
