def get_vit_scale(scale: str) -> tuple[int, int, int]:
    if scale == "tiny":
        return 192, 12, 3
    elif scale == "small":
        return 384, 12, 6
    elif scale == "base":
        return 768, 12, 12
    elif scale == "large":
        return 1024, 24, 16
    elif scale == "huge":
        return 1280, 32, 16
    else:
        raise KeyError(f"Unknown Vision Transformer scale: {scale}")

def get_global_attn_indexes(num_layers: int) -> list[int]:
    """
    Args:
        num_layers (int): The number of layers.

    Returns:
        List[int]: The global attention indexes.
    """

    return list(range(num_layers // 4 - 1, num_layers, num_layers // 4))
