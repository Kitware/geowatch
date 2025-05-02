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
    disallows de-serialization of unknown classes. Safely handles cases where
    attributes might not exist in different Python/NumPy versions.

    References:
        https://github.com/huggingface/transformers/pull/34632
    """
    import torch
    import kwcoco
    import numpy as np

    # Safely get multiarray module
    try:
        np_core = np._core  # Newer NumPy versions
    except AttributeError:
        np_core = np.core  # Older NumPy versions

    # Build safe globals list dynamically
    safe_globals = []

    # Add kwcoco.CategoryTree if available
    safe_globals.append(kwcoco.category_tree.CategoryTree)

    # Add numpy multiarray reconstruct if available
    safe_globals.append(np_core.multiarray._reconstruct)
    safe_globals.append(np_core.multiarray.scalar)

    # Add basic numpy types
    safe_globals.extend([
        np.ndarray,
        np.dtype,
    ])

    # Add numpy dtype classes if they exist
    dtype_classes = [
        'UInt8DType', 'UInt16DType', 'UInt32DType', 'UInt64DType',
        'Int8DType', 'Int16DType', 'Int32DType', 'Int64DType',
        'Float16DType', 'Float32DType', 'Float64DType',
        'Complex64DType', 'Complex128DType',
        'BoolDType'
    ]

    for dtype_name in dtype_classes:
        try:
            dtype_class = getattr(np.dtypes, dtype_name)
            safe_globals.append(dtype_class)
        except AttributeError:
            continue

    # Only try to add safe globals if we found any and torch supports it
    if safe_globals:
        try:
            torch.serialization.add_safe_globals(safe_globals)
        except AttributeError:
            pass
