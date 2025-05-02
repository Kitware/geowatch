def fix_gelu_issue(method):
    """
    Torch 1.12 added an approximate parameter that our old models dont have.
    Monkey patch it in.

    References:
        https://github.com/pytorch/pytorch/pull/61439
    """
    for name, mod in method.named_modules():
        if mod.__class__.__name__ == "GELU":
            if not hasattr(mod, "approximate"):
                mod.approximate = "none"


def fix_package_modules():
    # Monkey Patch torch.package
    import sys

    if sys.version_info[0:2] >= (3, 10):
        try:
            from torch.package import _stdlib

            _stdlib._get_stdlib_modules = lambda: sys.stdlib_module_names
        except Exception:
            pass


def add_safe_globals():
    """
    Stop gap to allow for loading of common classes now that torch load
    disallows de-serialization of unknown classes.

    References:
        https://github.com/huggingface/transformers/pull/34632
    """
    import torch
    import kwcoco
    import numpy as np

    try:
        ma = np._core.multiarray
    except AttributeError:
        ma = np.core.multiarray
    try:
        torch.serialization.add_safe_globals(
            [
                kwcoco.category_tree.CategoryTree,
                ma._reconstruct,
                np.ndarray,
                np.dtype,
                np.dtypes.UInt32DType,
            ]
        )
    except AttributeError:
        ...
