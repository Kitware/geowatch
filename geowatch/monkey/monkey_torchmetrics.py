def fix_torchmetrics_compatability():
    import torchmetrics
    from geowatch.monkey._monkey_fbeta import FBetaScore_Patched
    from geowatch.monkey._monkey_fbeta import Accuracy_Patched
    f_beta = torchmetrics.classification.f_beta
    accuracy = torchmetrics.classification.accuracy

    if not hasattr(f_beta, 'FBetaScoreOrig'):
        f_beta.FBetaScoreOrig = f_beta.FBetaScore

    if not hasattr(accuracy, 'FBetaScoreOrig'):
        accuracy.AccuracyOrig = accuracy.Accuracy

    f_beta.FBetaScore = FBetaScore_Patched
    f_beta.F1 = FBetaScore_Patched

    accuracy.Accuracy = Accuracy_Patched
