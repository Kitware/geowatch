"""
Lazilly loaded, but global module containing definitions for
monkey_torchmetrics
"""
import torchmetrics
from typing import Any, Optional
from typing_extensions import Literal
from torchmetrics.metric import Metric


class FBetaScore_Patched:
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
        f_beta = torchmetrics.classification.f_beta
        assert multidim_average is not None

        # This is the main monkey patch: setting task heuristically
        # because old models may not have had it.
        if task is None or task == 'FBetaScore()':
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


class Accuracy_Patched:
    r"""Computes `Accuracy`_
    """

    def __new__(
        cls,
        task: Literal["binary", "multiclass", "multilabel"] = None,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        multidim_average: Literal["global", "samplewise"] = "global",
        top_k: Optional[int] = 1,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        accuracy = torchmetrics.classification.accuracy

        if task is None or task == 'FBetaScore()':
            if num_classes is None:
                task = 'binary'
            else:
                task = 'multiclass'

        kwargs.update(dict(multidim_average=multidim_average, ignore_index=ignore_index, validate_args=validate_args))
        if task == "binary":
            return accuracy.BinaryAccuracy(threshold, **kwargs)
        if task == "multiclass":
            assert isinstance(num_classes, int)
            assert isinstance(top_k, int)
            return accuracy.MulticlassAccuracy(num_classes, top_k, average, **kwargs)
        if task == "multilabel":
            assert isinstance(num_labels, int)
            return accuracy.MultilabelAccuracy(num_labels, threshold, average, **kwargs)
        raise ValueError(
            f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
        )
