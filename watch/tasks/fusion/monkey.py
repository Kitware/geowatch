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
