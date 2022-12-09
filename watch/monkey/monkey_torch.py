def fix_gelu_issue(method):
    # Torch 1.12 added an approximate parameter that our old models dont
    # have. Monkey patch it in.
    # https://github.com/pytorch/pytorch/pull/61439
    for name, mod in method.named_modules():
        if mod.__class__.__name__ == 'GELU':
            if not hasattr(mod, 'approximate'):
                mod.approximate = 'none'
