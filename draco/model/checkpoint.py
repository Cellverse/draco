from pathlib import Path
from typing import Any

import torch


def _strip_prefix_if_present(state_dict: dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


def load_pretrained(model: torch.nn.Module, ckpt_path: Path, device: torch.device = "cuda") -> torch.nn.Module:
    """
    Load the pre-trained model from the checkpoint file.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "state_dict" in ckpt:
        checkpoint_state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        checkpoint_state_dict = ckpt["model"]
    else:
        checkpoint_state_dict = ckpt

    _strip_prefix_if_present(checkpoint_state_dict, "module.")    # for DistributedDataParallel
    _strip_prefix_if_present(checkpoint_state_dict, "model.")     # for PyTorch Lightning Module
    _strip_prefix_if_present(checkpoint_state_dict, "_orig_mod.") # for torch.compile

    msg = model.load_state_dict(checkpoint_state_dict, strict=False)
    print(f"Loaded pre-trained model from {ckpt_path} with message: {msg}")

    return model
