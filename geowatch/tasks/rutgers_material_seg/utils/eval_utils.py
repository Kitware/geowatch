import numpy as np
from torch import nn
import torch
import re
import math
import functools
from sklearn.metrics import average_precision_score


def get_ap_score(y_true, y_scores):
    """
    Get average precision score between 2 1-d numpy arrays

    Args:
        y_true: batch of true labels
        y_scores: batch of confidence scores

    Returns:
        sum of batch average precision
    """
    scores = average_precision_score(y_true=y_true, y_score=y_scores, average='samples')
    return scores


def val_mae(pred_counts, gt_counts):
    n = len(gt_counts)
    true_count = np.ones(n) * (-1)
    pred_count = np.ones(n) * (-1)

    for index in range(n):
        true_count[index] = gt_counts[index]
        pred_count[index] = pred_counts[index]
        # mae = (np.abs(true_count[:index+1] - pred_count[:index+1])).mean()
    score_dict = {}
    assert not np.any(true_count == (-1))
    assert not np.any(pred_count == (-1))
    score_dict["MAE"] = (np.abs(true_count - pred_count)).mean()
    print(f"score_dict: {score_dict}")
    return score_dict


def mae(pred_counts, gt_counts):
    assert len(pred_counts) == len(gt_counts)
    n = len(gt_counts)
    absolute_e_list = [abs(b_i - a_i) for a_i, b_i in zip(pred_counts, gt_counts)]
    sum_e_list = sum(absolute_e_list)
    mae = sum_e_list / n
    return mae


def rmse(pred_counts, gt_counts):
    assert len(pred_counts) == len(gt_counts)
    n = len(gt_counts)

    absolute_e_list = [abs(b_i - a_i) * abs(b_i - a_i) for a_i, b_i in zip(pred_counts, gt_counts)]
    sum_e_list = sum(absolute_e_list)
    sum_e_list = sum_e_list / n
    rmse = math.sqrt(sum_e_list)
    return rmse


def mape(pred_counts, gt_counts):
    assert len(pred_counts) == len(gt_counts)
    n = len(gt_counts)

    absolute_e_list = [abs(b_i - a_i) / b_i for a_i, b_i in zip(pred_counts, gt_counts)]
    sum_e_list = sum(absolute_e_list)
    sum_e_list = sum_e_list / n
    mape = 100 * sum_e_list
    return mape


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1])
                    if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels))
              for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou_npy(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    intersection = np.sum(gt * pr)
    print(intersection)
    union = np.sum(gt) + np.sum(pr) - intersection + eps
    print(union)
    return (intersection + eps) / union


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = (((1 + beta ** 2) * tp + eps) /
             ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps))

    return score


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr)
    score = tp / gt.view(-1).shape[0]
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score


class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name


class Activation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        if activation is None or activation == 'identity':
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'softmax2d':
            self.activation = functools.partial(torch.softmax, dim=1)
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError

    def forward(self, x):
        return self.activation(x)


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def calc_mAP(predictions, gts, num_classes=2):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) +
                          hist.sum(axis=0) - np.diag(hist))
    # iu = np.diag(hist) / (hist.sum(axis=1) +
    #                       - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


class Metric(BaseObject):
    pass


class IoU(Metric):
    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Fscore(Metric):

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


def compute_jaccard(preds_masks_all, targets_masks_all, num_classes=21):

    tps = np.zeros((num_classes, ))
    fps = np.zeros((num_classes, ))
    fns = np.zeros((num_classes, ))
    # counts = np.zeros((num_classes, ))

    for mask_pred, mask_gt in zip(preds_masks_all, targets_masks_all):
        # print(mask_pred.shape)
        # print(mask_gt.shape)
        # print(mask_pred.size())
        bs, h, w = mask_pred.size()
        assert bs == mask_gt.size(0), "Batch size mismatch"
        assert h == mask_gt.size(1), "Width mismatch"
        assert w == mask_gt.size(2), "Height mismatch"

        mask_pred = mask_pred.view(bs, 1, -1)
        mask_gt = mask_gt.view(bs, 1, -1)

        # ignore ambiguous
        # mask_pred[mask_gt == 255] = 255

        for label in range(num_classes):
            mask_pred_ = (mask_pred == label).float()
            mask_gt_ = (mask_gt == label).float()

            tps[label] += (mask_pred_ * mask_gt_).sum().item()
            diff = mask_pred_ - mask_gt_
            fps[label] += np.maximum(0., diff).float().sum().item()
            fns[label] += np.maximum(0., -diff).float().sum().item()

    eps = 1e-3
    jaccards  = [None] * num_classes
    precision = [None] * num_classes
    recall    = [None] * num_classes
    for i in range(num_classes):
        tp = tps[i]
        fn = fns[i]
        fp = fps[i]
        jaccards[i]  = tp / max(eps, fn + fp + tp)
        precision[i] = tp / max(eps, tp + fp)
        recall[i]    = tp / max(eps, tp + fn)

    return jaccards, precision, recall
