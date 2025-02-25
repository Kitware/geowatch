import torch


def ordinal_position_encoding(num_items, feat_size, method='sin', device='cpu'):
    """
    A positional encoding that represents ordinal

    Args:
        num_items (int): number of dimensions to be encoded (
            e.g. this is a spatial or temporal index)
        feat_size (int): this is the number of dimensions in the positional
             encoding generated for each dimension / item

    Example:
        >>> # Use 5 feature dimensions to encode 3 timesteps
        >>> from geowatch.tasks.fusion.datamodules.util_positional_encoding import *  # NOQA
        >>> num_timesteps = num_items = 3
        >>> feat_size = 5
        >>> encoding = ordinal_position_encoding(num_items, feat_size)
    """
    assert method == 'sin'
    sf = 10000
    parts = []
    base = torch.arange(num_items, device=device)
    for idx in range(feat_size):
        exponent = (idx / feat_size)
        modulator = (1 / (sf ** exponent))
        theta = base * modulator
        if idx % 2 == 0:
            part = torch.sin(theta)
        else:
            part = torch.cos(theta)
        parts.append(part)
    encoding = torch.stack(parts, dim=1)
    return encoding
