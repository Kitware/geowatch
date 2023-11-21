import torchmetrics

import warnings
warnings.warn('''
    util_torchmetrics is deprecated since torchmetrics==0.10.0.
    Consider replacing calls to `util_torchmetrics.Binary{Metric}(...)`
    with calls to `torchmetrics.{Metric}(..., task='binary')`.
''', DeprecationWarning, stacklevel=2)

'''
This is a stopgap measure until https://github.com/Lightning-AI/metrics/pull/1195 is integrated
integrated into torchmetrics (targeted for v0.10?). The main issue that these metrics solve is
that when computing classification stats for *binary* problems, the returned score is an average
over the two classes stats when most of the time we only want one of them. This is especially
necessary in imbalanced settings where the metric for the largest class completely obscures the
impact of the smaller class.

SeeAlso:
    ../monkey/monkey_torchmetrics.py
'''


class BinaryF1Score(torchmetrics.StatScores):
    def compute(self):
        tp, fp, tn, fn = self._get_final_stats()
        tp, fp, tn, fn = tp[1], fp[1], tn[1], fn[1]
        return (2 * tp) / ((2 * tp) + fp + fn)


class BinaryOverallAccuracy(torchmetrics.StatScores):
    def compute(self):
        tp, fp, tn, fn = self._get_final_stats()
        tp, fp, tn, fn = tp[1], fp[1], tn[1], fn[1]
        return (tp + tn) / (tp + fp + tn + fn)


class BinaryBalancedAccuracy(torchmetrics.StatScores):
    def compute(self):
        tp, fp, tn, fn = self._get_final_stats()
        tp, fp, tn, fn = tp[1], fp[1], tn[1], fn[1]

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)

        return (tpr + tnr) / 2


class BinaryTruePositiveRate(torchmetrics.StatScores):
    def compute(self):
        tp, fp, tn, fn = self._get_final_stats()
        tp, fp, tn, fn = tp[1], fp[1], tn[1], fn[1]

        return tp / (tp + fn)
