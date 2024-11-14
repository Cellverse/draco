from functools import partial
from typing import Any, Callable

from omegaconf import DictConfig
from timm.layers import build_sincos2d_pos_embed, resample_abs_pos_embed_nhwc, PatchEmbed, Mlp, LayerType
from timm.models.vision_transformer import Block
from timm.models.vision_transformer_sam import Block as SAMBlock
import torch
import torch.nn as nn

from draco.configuration import configurable
from .build import MODEL_REGISTRY
from .layer import LayerNorm2d
from .draco_base import DenoisingReconstructionAutoencoderVisionTransformerBase
from .utils.constant import get_vit_scale, get_global_attn_indexes

__all__ = ["DenoisingReconstructionAutoencoderVisionTransformer2d", "DracoDenoiseAutoencoder"]


@MODEL_REGISTRY.register()
class DenoisingReconstructionAutoencoderVisionTransformer2d(DenoisingReconstructionAutoencoderVisionTransformerBase):
    @configurable
    def __init__(self, *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_layer: Callable = PatchEmbed,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        use_abs_pos: bool = True,
        block_fn: nn.Module = Block,
        norm_layer: LayerType = partial(nn.LayerNorm, eps=1e-6),
        act_layer: LayerType = nn.GELU,
        mlp_layer: nn.Module = Mlp,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        decoder_block_fn: nn.Module = Block,
        decoder_norm_layer: LayerType = partial(nn.LayerNorm, eps=1e-6),
        decoder_act_layer: LayerType = nn.GELU,
        decoder_mlp_layer: nn.Module = Mlp,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        decoder_use_neck: bool = True,
        decoder_neck_dim: int = 256,
    ) -> None:
        super().__init__()

        self.dynamic_img_size = dynamic_img_size
        self.decoder_use_neck = decoder_use_neck

        self.init_encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_layer=embed_layer,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            use_abs_pos=use_abs_pos,
            block_fn=block_fn,
            norm_layer=norm_layer,
            act_layer=act_layer,
            mlp_layer=mlp_layer,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.init_decoder(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_abs_pos=use_abs_pos,
            decoder_block_fn=decoder_block_fn,
            decoder_norm_layer=decoder_norm_layer,
            decoder_act_layer=decoder_act_layer,
            decoder_mlp_layer=decoder_mlp_layer,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            decoder_use_neck=decoder_use_neck,
            decoder_neck_dim=decoder_neck_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.init_weights(
            grid_size=self.patch_embed.grid_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        embed_dim, depth, num_heads = get_vit_scale(cfg.MODEL.VIT_SCALE)
        return {
            "img_size": cfg.MODEL.IMG_SIZE,
            "patch_size": cfg.MODEL.PATCH_SIZE,
            "in_chans": cfg.MODEL.IN_CHANS,
            "dynamic_img_size": cfg.MODEL.DYNAMIC_IMG_SIZE,
            "dynamic_img_pad": cfg.MODEL.DYNAMIC_IMG_PAD,
            "use_abs_pos": cfg.MODEL.USE_ABS_POS,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "decoder_embed_dim": cfg.MODEL.DECODER_EMBED_DIM,
            "decoder_depth": cfg.MODEL.DECODER_DEPTH,
            "decoder_num_heads": cfg.MODEL.DECODER_NUM_HEADS,
            "decoder_use_neck": cfg.MODEL.DECODER_USE_NECK,
            "decoder_neck_dim": cfg.MODEL.DECODER_NECK_DIM,
        }

    def init_encoder(self, *,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_layer: Callable,
        dynamic_img_size: bool,
        dynamic_img_pad: bool,
        use_abs_pos: bool,
        block_fn: nn.Module,
        norm_layer: LayerType | None,
        act_layer: LayerType | None,
        mlp_layer: nn.Module,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_norm: bool,
    ) -> None:
        embed_args = {}
        if dynamic_img_size:
            embed_args.update(dict(strict_img_size=False))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            dynamic_img_pad=dynamic_img_pad,
            output_fmt="NHWC",
            **embed_args
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, *self.patch_embed.grid_size, embed_dim)) if use_abs_pos else None
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            ) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def init_decoder(self, *,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        use_abs_pos: bool,
        decoder_block_fn: nn.Module,
        decoder_norm_layer: LayerType | None,
        decoder_act_layer: LayerType | None,
        decoder_mlp_layer: nn.Module,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        decoder_use_neck: bool,
        decoder_neck_dim: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_norm: bool,
    ) -> None:
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, *self.patch_embed.grid_size, decoder_embed_dim)) if use_abs_pos else None
        self.decoder_blocks = nn.ModuleList([
            decoder_block_fn(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=decoder_norm_layer,
                act_layer=decoder_act_layer,
                mlp_layer=decoder_mlp_layer,
            ) for _ in range(decoder_depth)
        ])
        self.decoder_norm = decoder_norm_layer(decoder_embed_dim)
        if decoder_use_neck:
            self.decoder_neck = nn.Sequential(
                nn.Conv2d(
                    in_channels=decoder_embed_dim,
                    out_channels=decoder_neck_dim,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(decoder_neck_dim),
                decoder_act_layer(),
                nn.Conv2d(
                    in_channels=decoder_neck_dim,
                    out_channels=decoder_neck_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(decoder_neck_dim),
                decoder_act_layer(),
                nn.Conv2d(
                    in_channels=decoder_neck_dim,
                    out_channels=decoder_embed_dim,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(decoder_embed_dim),
            )
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans)

    def init_weights(self, *,
        grid_size: tuple[int, int],
        embed_dim: int,
        decoder_embed_dim: int
    ) -> None:
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view(w.size(0), -1))

        torch.nn.init.normal_(self.mask_token, std=0.02)

        if self.pos_embed is not None:
            self.pos_embed.data.copy_(build_sincos2d_pos_embed(
                feat_shape=grid_size,
                dim=embed_dim,
                interleave_sin_cos=True
            ).reshape(1, *grid_size, -1).transpose(1, 2))

        if self.decoder_pos_embed is not None:
            self.decoder_pos_embed.data.copy_(build_sincos2d_pos_embed(
                feat_shape=grid_size,
                dim=decoder_embed_dim,
                interleave_sin_cos=True
            ).reshape(1, *grid_size, -1).transpose(1, 2))

        if self.decoder_use_neck:
            for m in self.decoder_neck.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.zeros_(self.decoder_neck[-1].weight)
            nn.init.zeros_(self.decoder_neck[-1].bias)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.BoolTensor, int, int]:
        x = self.patch_embed(x)
        B, H, W, E = x.shape
        if self.pos_embed is not None:
            x = x + resample_abs_pos_embed_nhwc(self.pos_embed, (H, W))
        x = x.view(B, -1, E)

        mask = super().random_masking(x, mask_ratio)
        x = x[~mask].reshape(B, -1, E)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x, mask, H, W

    def forward_decoder(self, x: torch.Tensor, mask: torch.BoolTensor, H: int, W: int) -> torch.Tensor:
        x = self.decoder_embed(x)

        B, L = mask.shape
        E = x.shape[-1]
        mask_tokens = self.mask_token.repeat(B, L, 1).to(x.dtype)
        mask_tokens[~mask] = x.reshape(-1, E)
        x = mask_tokens

        if self.decoder_pos_embed is not None:
            x = x.view(B, H, W, E)
            x = x + resample_abs_pos_embed_nhwc(self.decoder_pos_embed, (H, W))
            x = x.view(B, -1, E)

        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        if self.decoder_use_neck:
            x = x + self.decoder_neck(
                x.permute(0, 2, 1).reshape(B, E, H, W).contiguous()
            ).permute(0, 2, 3, 1).reshape(B, L, -1).contiguous()
        x = self.decoder_pred(x)

        return x

    def forward(self, x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.BoolTensor]:
        x, mask, H, W = self.forward_encoder(x, mask_ratio)
        x = self.forward_decoder(x, mask, H, W)
        return x, mask
    
@MODEL_REGISTRY.register()
class DracoDenoiseAutoencoder(DenoisingReconstructionAutoencoderVisionTransformerBase):
    """
    Masked Autoencoder (MAE) with Vision Transformer backbone.
    Note that `cls_token` is discarded.
    """

    @configurable
    def __init__(self, *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_layer: Callable = PatchEmbed,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        use_abs_pos: bool = True,
        block_fn: nn.Module = SAMBlock,
        norm_layer: LayerType = partial(nn.LayerNorm, eps=1e-6),
        act_layer: LayerType = nn.GELU,
        mlp_layer: nn.Module = Mlp,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        window_size: int = 16,
        global_attn_indexes: list[int] = [2, 5, 8, 11],
        decoder_block_fn: nn.Module = SAMBlock,
        decoder_norm_layer: LayerType = partial(nn.LayerNorm, eps=1e-6),
        decoder_act_layer: LayerType = nn.GELU,
        decoder_mlp_layer: nn.Module = Mlp,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        decoder_use_neck: bool = True,
        decoder_neck_dim: int = 256,
        decoder_global_attn_indexes: list[int] = [3, 7],
    ) -> None:
        super().__init__()

        self.dynamic_img_size = dynamic_img_size
        self.decoder_use_neck = decoder_use_neck

        self.init_encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_layer=embed_layer,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            use_abs_pos=use_abs_pos,
            block_fn=block_fn,
            norm_layer=norm_layer,
            act_layer=act_layer,
            mlp_layer=mlp_layer,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            window_size=window_size,
            global_attn_indexes=global_attn_indexes
        )
        self.init_decoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_abs_pos=use_abs_pos,
            decoder_block_fn=decoder_block_fn,
            decoder_norm_layer=decoder_norm_layer,
            decoder_act_layer=decoder_act_layer,
            decoder_mlp_layer=decoder_mlp_layer,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            decoder_use_neck=decoder_use_neck,
            decoder_neck_dim=decoder_neck_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            window_size=window_size,
            decoder_global_attn_indexes=decoder_global_attn_indexes
        )
        self.init_weights(
            grid_size=self.patch_embed.grid_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
        )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        embed_dim, depth, num_heads = get_vit_scale(cfg.MODEL.VIT_SCALE)
        global_attn_indexes = get_global_attn_indexes(depth)
        return {
            "img_size": cfg.MODEL.IMG_SIZE,
            "patch_size": cfg.MODEL.PATCH_SIZE,
            "in_chans": cfg.MODEL.IN_CHANS,
            "dynamic_img_size": cfg.MODEL.DYNAMIC_IMG_SIZE,
            "dynamic_img_pad": cfg.MODEL.DYNAMIC_IMG_PAD,
            "use_abs_pos": cfg.MODEL.USE_ABS_POS,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "window_size": cfg.MODEL.WINDOW_SIZE,
            "global_attn_indexes": global_attn_indexes,
            "decoder_embed_dim": cfg.MODEL.DECODER_EMBED_DIM,
            "decoder_depth": cfg.MODEL.DECODER_DEPTH,
            "decoder_num_heads": cfg.MODEL.DECODER_NUM_HEADS,
            "decoder_use_neck": cfg.MODEL.DECODER_USE_NECK,
            "decoder_neck_dim": cfg.MODEL.DECODER_NECK_DIM,
            "decoder_global_attn_indexes": cfg.MODEL.DECODER_GLOBAL_ATTN_INDEXES,
        }

    def init_encoder(self, *,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_layer: Callable,
        dynamic_img_size: bool,
        dynamic_img_pad: bool,
        use_abs_pos: bool,
        block_fn: nn.Module,
        norm_layer: LayerType | None,
        act_layer: LayerType | None,
        mlp_layer: nn.Module,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_norm: bool,
        window_size: int,
        global_attn_indexes: list,
    ) -> None:
        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            dynamic_img_pad=dynamic_img_pad,
            output_fmt="NHWC",
            **embed_args
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, *self.patch_embed.grid_size, embed_dim)) if use_abs_pos else None
        self.blocks = nn.ModuleList(
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                use_rel_pos=True,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            ) for i in range(depth)
        )

        self.norm = norm_layer(embed_dim)

    def init_decoder(self, *,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        use_abs_pos: bool,
        decoder_block_fn: nn.Module,
        decoder_norm_layer: LayerType | None,
        decoder_act_layer: LayerType | None,
        decoder_mlp_layer: nn.Module,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        decoder_use_neck: bool,
        decoder_neck_dim: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_norm: bool,
        window_size: int,
        decoder_global_attn_indexes: list[int]
    ) -> None:
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, *self.patch_embed.grid_size, decoder_embed_dim)) if use_abs_pos else None
        self.decoder_blocks = nn.ModuleList(
            decoder_block_fn(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                norm_layer=decoder_norm_layer,
                act_layer=decoder_act_layer,
                mlp_layer=decoder_mlp_layer,
                use_rel_pos=True,
                window_size=window_size if i not in decoder_global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            ) for i in range(decoder_depth)
        )
        self.decoder_norm = decoder_norm_layer(decoder_embed_dim)
        if decoder_use_neck:
            self.decoder_neck = nn.Sequential(
                nn.Conv2d(
                    in_channels=decoder_embed_dim,
                    out_channels=decoder_neck_dim,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(decoder_neck_dim),
                decoder_act_layer(),
                nn.Conv2d(
                    in_channels=decoder_neck_dim,
                    out_channels=decoder_neck_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(decoder_neck_dim),
                decoder_act_layer(),
                nn.Conv2d(
                    in_channels=decoder_neck_dim,
                    out_channels=decoder_embed_dim,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(decoder_embed_dim),
            )
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans)

    def init_weights(self, *,
        grid_size: tuple[int, int],
        embed_dim: int,
        decoder_embed_dim: int
    ) -> None:
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view(w.size(0), -1))

        if self.pos_embed is not None:
            self.pos_embed.data.copy_(build_sincos2d_pos_embed(
                feat_shape=grid_size,
                dim=embed_dim,
                interleave_sin_cos=True
            ).reshape(1, *grid_size, -1).transpose(1, 2))

        if self.decoder_pos_embed is not None:
            self.decoder_pos_embed.data.copy_(build_sincos2d_pos_embed(
                feat_shape=grid_size,
                dim=decoder_embed_dim,
                interleave_sin_cos=True
            ).reshape(1, *grid_size, -1).transpose(1, 2))

        # Zero-initialize the neck
        if self.decoder_use_neck:
            for m in self.decoder_neck.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.zeros_(self.decoder_neck[-1].weight)
            nn.init.zeros_(self.decoder_neck[-1].bias)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Forward pass of the encoder.

        Args:
            `x` (torch.Tensor): Image of shape [B, C, H, W].

        Returns:
            (torch.Tensor): Encoded image of shape [B, num_kept, E].
            (int): Height of the encoded tokens.
            (int): Width of the encoded tokens.
        """
        x = self.patch_embed(x)
        B, H, W, E = x.shape

        if self.pos_embed is not None:
            x = x + resample_abs_pos_embed_nhwc(self.pos_embed, (H, W))

        for block in self.blocks:
            x = block(x)

        x = x.view(B, -1, E)
        x = self.norm(x)

        return x, H, W

    def forward_decoder(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            `x` (torch.Tensor): Encoded image of shape [B, num_kept, E].
            `H` (int): Height of the encoded tokens.
            `W` (int): Width of the encoded tokens.

        Returns:
            (torch.Tensor): Decoded image of shape [B, L, E].
        """
        x = self.decoder_embed(x) # [B, num_kept, E]
        B, L, E = x.shape

        if self.decoder_pos_embed is not None:
            x = x.view(B, H, W, E)
            x = x + resample_abs_pos_embed_nhwc(self.decoder_pos_embed, (H, W))

        for block in self.decoder_blocks:
            x = block(x)
        x = x.view(B, -1, E)

        x = self.decoder_norm(x)
        if self.decoder_use_neck:
            x = x + self.decoder_neck(
                x.permute(0, 2, 1).reshape(B, E, H, W).contiguous()
            ).permute(0, 2, 3, 1).reshape(B, L, -1).contiguous()
        x = self.decoder_pred(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            `x` (torch.Tensor): Image of shape [B, C, H, W].

        Returns:
            (torch.Tensor): The prediction of shape [B, L, E].
        """
        x, H, W = self.forward_encoder(x)
        x = self.forward_decoder(x, H, W)
        return x

