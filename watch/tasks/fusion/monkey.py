import torchmetrics
if not hasattr(torchmetrics.classification.f_beta, 'F1'):
    torchmetrics.classification.f_beta.F1 = torchmetrics.classification.f_beta.FBetaScore


def torchmetrics_compat_hack():
    import torchmetrics
    f_beta = torchmetrics.classification.f_beta
    if not hasattr(f_beta, 'FBeta'):
        f_beta.FBeta = f_beta.FBetaScore
    if not hasattr(torchmetrics.classification.f_beta, 'FBetaScore'):
        f_beta.FBetaScore = f_beta.FBeta


torchmetrics_compat_hack()


def fix_gelu_issue(method):
    # Torch 1.12 added an approximate parameter that our old models dont
    # have. Monkey patch it in.
    # https://github.com/pytorch/pytorch/pull/61439
    for name, mod in method.named_modules():
        if mod.__class__.__name__ == 'GELU':
            if not hasattr(mod, 'approximate'):
                mod.approximate = 'none'


# Also one in:
# from watch.utils.lightning_ext.callbacks.packager import _torch_package_monkeypatch
