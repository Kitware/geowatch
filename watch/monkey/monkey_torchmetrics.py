def fix_torchmetrics_compatability():
    import torchmetrics

    if not hasattr(torchmetrics.classification.f_beta, 'F1'):
        torchmetrics.classification.f_beta.F1 = torchmetrics.classification.f_beta.FBetaScore

    from watch.monkey._monkey_fbeta import FBetaScore_Patched
    f_beta = torchmetrics.classification.f_beta

    if not hasattr(torchmetrics.classification.f_beta, 'FBetaScoreOrig'):
        f_beta.FBetaScoreOrig = f_beta.FBetaScore

    # if hasattr(torchmetrics.classification.f_beta, 'FBetaScore'):
    #     if not hasattr(f_beta, 'FBeta'):
    #         f_beta.FBeta = f_beta.FBetaScore
    # if hasattr(f_beta, 'FBeta'):
    #     if not hasattr(torchmetrics.classification.f_beta, 'FBetaScore'):
    #         f_beta.FBetaScore = f_beta.FBeta

    # def _FBetaScore_HackedSignature(task=None, beta=1.0, threshold=0.5,
    #                                 num_classes=None, num_labels=None,
    #                                 average='micro', multidim_average='global',
    #                                 top_k=1, ignore_index=None,
    #                                 validate_args=True, **kwargs):
    # def _FBetaScore_HackedSignature(**kwargs):
    #     task = kwargs.get('task', None)
    #     num_classes = kwargs.get('num_classes', None)
    #     if task is None:
    #         if num_classes is None:
    #             kwargs['task'] = 'binary'
    #         else:
    #             kwargs['task'] = 'multiclass'
    #     return f_beta.FBetaScoreOrig(**kwargs)
    f_beta.FBetaScore = FBetaScore_Patched
