import torchmetrics
from typing import Any, Optional
from typing_extensions import Literal
from torchmetrics.metric import Metric

if not hasattr(torchmetrics.classification.f_beta, 'F1'):
    torchmetrics.classification.f_beta.F1 = torchmetrics.classification.f_beta.FBetaScore


def torchmetrics_compat_hack():
    import torchmetrics
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

    f_beta.FBetaScore = FBetaScoreHacked


class FBetaScoreHacked:
    r"""Computes `F-score`_ metric:

    .. math::
        F_{\beta} = (1 + \beta^2) * \frac{\text{precision} * \text{recall}}
        {(\beta^2 * \text{precision}) + \text{recall}}

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`binary_fbeta_score`, :func:`multiclass_fbeta_score` and :func:`multilabel_fbeta_score` for the specific
    details of each argument influence and examples.
    """

    def __new__(
        cls,
        task: Literal["binary", "multiclass", "multilabel"] = None,
        beta: float = 1.0,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        multidim_average: Optional[Literal["global", "samplewise"]] = "global",
        top_k: Optional[int] = 1,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        import torchmetrics
        f_beta = torchmetrics.classification.f_beta
        assert multidim_average is not None
        if task is None:
            if num_classes is None:
                task = 'binary'
            else:
                task = 'multiclass'
        kwargs.update(dict(multidim_average=multidim_average, ignore_index=ignore_index, validate_args=validate_args))
        if task == "binary":
            return f_beta.BinaryFBetaScore(beta, threshold, **kwargs)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            assert isinstance(top_k, int)
            return f_beta.MulticlassFBetaScore(beta, num_classes, top_k, average, **kwargs)
        if task == "multilabel":
            assert isinstance(num_labels, int)
            return f_beta.MultilabelFBetaScore(beta, num_labels, threshold, average, **kwargs)
        raise ValueError(
            f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
        )


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
