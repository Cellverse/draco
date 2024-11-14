from fvcore.common.registry import Registry
from omegaconf import DictConfig
import torch

__all__ = ["MODEL_REGISTRY", "build_model"]


MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = "Registry for the model."

def build_model(cfg: DictConfig) -> torch.nn.Module:
    """
    Build the model defined by `cfg.MODEL.NAME`.
    It moves the model to the device defined by `cfg.MODEL.DEVICE`.
    It does not load checkpoints from `cfg`.
    """
    model_name = cfg.MODEL.NAME
    try:
        model = MODEL_REGISTRY.get(model_name)(cfg)
    except KeyError as e:
        raise KeyError(MODEL_REGISTRY) from e
    return model
