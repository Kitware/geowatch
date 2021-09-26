# import warnings
# from typing import Optional, Sequence, Union
import torch
# import torch.nn.functional as F
# from torch.nn.modules.loss import _Loss
# from monai.networks import one_hot
# from monai.utils import LossReduction


class CrossEntropyLoss_KeepDim(torch.nn.CrossEntropyLoss):
    def forward(self, input, target):
        ce_loss = super().forward(input, target)
        if self.reduction == 'none':
            # Add in class dimension for broadcasting
            ce_loss = ce_loss[:, None]
        return ce_loss


class BCEWithLogitsLoss_BetterWeight(torch.nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        fixed_input = input.swapaxes(1, -1)
        fixed_target = target.swapaxes(1, -1)
        loss = super().forward(fixed_input, fixed_target)
        if self.reduction == 'none':
            loss = loss.swapaxes(1, -1)
        return loss


def inverse_sigmoid(x):
    return -torch.log((1 / (x + 1e-8)) - 1)


def single_example_dice_focal():
    import kwimage
    import monai
    import torch
    import numpy as np
    import kwarray

    S = 64
    poly = kwimage.Polygon.random().scale(S).scale(1.1, about='center')

    is_fg = np.zeros((S, S))
    is_fg = poly.fill(is_fg, value=1).astype(np.int64)
    is_bg = (1 - is_fg)

    C = 3

    eps = 1e-3
    dist_matrix = 1 - np.eye(C, C)

    true_cxs = torch.Tensor(is_fg[None, :]).long()
    true_ohe = kwarray.one_hot_embedding(true_cxs, C)

    prob = np.zeros_like(is_fg)

    realpos_mean = np.linspace(eps, 1.0 - eps, 32)
    realneg_mean = np.linspace(eps, 1.0 - eps, 32)

    # rng = np.random.RandomState(0)

    # noise = rng.randn(*true_cxs.shape)
    noise = 0

    # Different loss criterions require different encodings of the truth
    targets = {
        'ohe': true_ohe,  # one hot embedding
        'idx': true_cxs,  # true class index
    }

    # Invert probabilities into raw network "logit" outputs

    class_weights = torch.Tensor([0.05, 5.0])

    loss_infos = [
        {'cls': monai.losses.FocalLoss, 'input_style': 'logit', 'target_style': 'ohe', 'kwargs': {'gamma': 5.0, 'weight': class_weights}},
        # {'cls': FixedFocalLoss, 'input_style': 'logit', 'target_style': 'ohe'},
        # {'cls': NetharnFocalLoss, 'input_style': 'logit', 'target_style': 'idx'},
        {'cls': monai.losses.DiceLoss, 'input_style': 'prob', 'target_style': 'ohe', 'kwargs': {'pixelwise': False}},
        {'cls': monai.losses.DiceFocalLoss, 'input_style': 'prob', 'target_style': 'ohe'},
        # {'cls': monai.losses.GeneralizedDiceLoss, 'input_style': 'prob', 'target_style': 'ohe'},
        # {'cls': monai.losses.GeneralizedWassersteinDiceLoss, 'input_style': 'prob', 'target_style': 'idx', 'kwargs': {'dist_matrix': dist_matrix}},
        # {'cls': monai.losses.DiceFocalLoss, 'input_style': 'logit', 'target_style': 'ohe'},
        # {'cls': torch.nn.CrossEntropyLoss, 'input_style': 'logit', 'target_style': 'idx'},

        {'cls': CrossEntropyLoss_KeepDim, 'input_style': 'logit', 'target_style': 'idx', 'kwargs': {'weight': class_weights}},

        # torch.nn.NLLLoss,
        # torch.nn.BCELoss,

        {'cls': BCEWithLogitsLoss_BetterWeight, 'input_style': 'logit', 'target_style': 'ohe', 'kwargs': {'pos_weight': class_weights}},
    ]

    for loss_info in loss_infos:
        kwargs = loss_info.get('kwargs', {})
        loss_info['instance'] = loss_info['cls'](reduction='mean', **kwargs)
        loss_info['name'] = loss_info['cls'].__name__

    # torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.ones(5))(torch.rand(3, 5), torch.rand(3, 5))
    # BCEWithLogitsLoss_BetterWeight(reduction='none', pos_weight=torch.ones(5))(torch.rand(3, 5, 1), torch.rand(3, 5, 1))
    # BCEWithLogitsLoss_BetterWeight(pos_weight=torch.ones(5))(torch.rand(3, 5, 1), torch.rand(3, 5, 1))

    def index_sample(total, num):
        if num >= total:
            return list(range(total))
        else:
            step = total // num
            return list(range(*slice(None, None, step).indices(total)))

    # Show modulated truth
    modulated_preds = []
    rows = []

    p_fg = 0.4
    p_bg = 0.3

    select_idxs = index_sample(len(realpos_mean), 3)
    select_jdxs = index_sample(len(realneg_mean), 3)

    def make_prob(p_fg, p_bg):
        fg_probs = ((p_fg + noise) * is_fg).clip(0, 1)
        bg_probs = ((p_bg + noise) * is_bg).clip(0, 1)
        other_probs = np.zeros_like(fg_probs)
        prob = np.stack([bg_probs, fg_probs, other_probs], axis=0)
        # Ensure sum to 1?
        prob[2] = 1 - prob.sum(axis=0)

        # prob = (torch.Tensor(prob) * 1).softmax(dim=0)
        # bg_probs[is_fg.astype(bool)] = 1 - fg_probs[is_fg.astype(bool)]
        # fg_probs[is_bg.astype(bool)] = 1 - bg_probs[is_bg.astype(bool)]
        prob = torch.Tensor(prob).clamp(eps, 1 - eps)[None, :]
        return prob

    def false_color(prob):
        pass

    rng = np.random.RandomState(0)
    idx_to_color = np.array(kwimage.Color.distinct(C))
    # rand_colors = rng.rand(C, 3)
    # seedmat = rng.rand(C, 3)
    # q, r = np.linalg.qr(seedmat)
    random_ortho = r
    # false_colored = (prob_hwc @ random_ortho)
    # false_colored = kwimage.normalize_intensity(false_colored)
    # u, s, vh = np.linalg.svd(seedmat, full_matrices=False)
    # print('u.shape = {!r}'.format(u.shape))
    # print('s.shape = {!r}'.format(s.shape))
    # print('vh.shape = {!r}'.format(vh.shape))
    # u @ np.diag(s) @ vh
    # ortho = u @ vh?

    import kwplot
    sns = kwplot.autosns()
    pnum_ = kwplot.PlotNums(nRows=len(select_idxs), nCols=len(select_jdxs))
    kwplot.figure(fnum=2)
    for p_fg in realpos_mean[select_idxs]:
        for p_bg in realneg_mean[select_jdxs]:
            prob_bchw = make_prob(p_fg, p_bg)
            import einops
            prob_hwc = einops.rearrange(prob_bchw[0], 'c h w -> h w c').numpy()
            prob_dims = prob_hwc.shape[0:2]

            chan_alphas = []
            for cx, prob_hw in enumerate(prob_bchw[0]):
                rgb_chan = np.tile(idx_to_color[cx][None, None, :], prob_dims + (1,))
                alpha_chan = prob_hw[:, :, None] * 0.5
                class_heatmap = np.concatenate([rgb_chan, alpha_chan], axis=2)
                chan_alphas.append(class_heatmap)

            background = kwimage.ensure_alpha_channel(np.zeros(prob_dims + (3,)))
            chan_alphas.append(background)
            colored_alpha = kwimage.overlay_alpha_layers(chan_alphas)
            # colored = kwimage.ensure_uint255(colored_alpha[:, :, 0:3])
            colored = colored_alpha

            # _, ax = kwplot.imshow(false_colored)
            fig = kwplot.figure(pnum=pnum_())
            # img_dims = prob_hwc.shape[0:2]
            # canvas = np.zeros(img_dims + (3,))
            # heatmap = kwimage.Heatmap(class_probs=prob_hwc, classes=['fg', 'bg'], img_dims=img_dims)
            # heatmap.colorize(with_alpha=False, imgspace=False, cmap='plasma').shape
            # kwimage.Heatmap(class_probs=prob_hwc).colorize().shape
            # heatmap.draw_on(canvas, imgspace=False)
            ax = fig.gca()
            kwplot.imshow(colored)
            title = f'p_fg={p_fg:0.3f}, p_bg={p_bg:0.3f}'
            ax.set_title(title)

    for idx, p_fg in enumerate(realpos_mean):
        for jdx, p_bg in enumerate(realneg_mean):
            prob = make_prob(p_fg, p_bg)
            fg_probs = ((p_fg + noise) * is_fg).clip(0, 1)
            bg_probs = ((p_bg + noise) * is_bg).clip(0, 1)
            # Ensure sum to 1
            bg_probs[is_fg.astype(bool)] = 1 - fg_probs[is_fg.astype(bool)]
            fg_probs[is_bg.astype(bool)] = 1 - bg_probs[is_bg.astype(bool)]
            prob = np.stack([bg_probs, fg_probs], axis=0)
            prob = torch.Tensor(prob).clamp(eps, 1 - eps)[None, :]

            logit = inverse_sigmoid(prob)

            inputs = {
                'logit': logit,
                'prob': prob,
                'p_fg': p_fg,
                'p_bg': p_bg,
            }

            modulated_preds.append(inputs)

            for loss_info in loss_infos:
                input = inputs[loss_info['input_style']]
                target = targets[loss_info['target_style']]
                loss_instance = loss_info['instance']
                loss_values = loss_instance(input, target).mean()

                loss = loss_values.item()
                rows.append({
                    'loss': loss,
                    'realpos_prob': p_fg,
                    'realneg_prob': p_bg,
                    'type': loss_info['name']
                })

    import pandas as pd
    df = pd.DataFrame(rows)
    import kwplot
    sns = kwplot.autosns()
    groups = dict(list(df.groupby('type')))
    pnum_ = kwplot.PlotNums(nSubplots=len(groups) + 1)
    fig = kwplot.figure(fnum=1, doclf=True)

    kwplot.figure(fnum=1, pnum=pnum_())
    ax = kwplot.imshow(is_fg)[1]
    ax.set_title('Truth (to be modulated)')

    for loss_name, group in groups.items():
        heatmap = group.pivot('realneg_prob', 'realpos_prob', 'loss')
        kwplot.figure(fnum=1, pnum=pnum_())
        ax = sns.heatmap(
            heatmap,
            robust=True,
            # vmin=0,
            # vmax=heatmap.values.max(),
            # xticklabels=['{:.2f}'.format(x) for x in heatmap.index.values],
            # yticklabels=['{:.2f}'.format(x) for x in heatmap.index.values],
            xticklabels=10,
            yticklabels=10,
        )
        ax.set_title(loss_name)

        new_labels = []
        for t in ax.get_xticklabels():
            new_labels.append(t.get_text()[0:4])
        ax.set_xticklabels(new_labels)

        new_labels = []
        for t in ax.get_yticklabels():
            new_labels.append(t.get_text()[0:4])
        ax.set_yticklabels(new_labels)

    fig.tight_layout()


# def test_loss():
#     import monai
#     import torch
#     import numpy as np
#     import kwarray
#     import einops
#     import ubelt as ub


#     if 0:
#         B = 101  # batch size
#         C = 7    # number of classes
#         H = 13   # 2D height
#         W = 17   # 2D width
#     else:
#         B = 30   # batch size
#         C = 2   # number of classes
#         H = 1   # 2D height
#         W = 1  # 2D width

#     eps = 1e-6
#     dist_matrix = 1 - np.eye(C, C)

#     loss_infos = [
#         {'cls': monai.losses.FocalLoss, 'input_style': 'logit', 'target_style': 'ohe', 'kwargs': {'gamma': 2.0}},
#         # {'cls': FixedFocalLoss, 'input_style': 'logit', 'target_style': 'ohe'},
#         # {'cls': NetharnFocalLoss, 'input_style': 'logit', 'target_style': 'idx'},
#         {'cls': monai.losses.DiceLoss, 'input_style': 'prob', 'target_style': 'ohe', 'kwargs': {'pixelwise': True}},
#         {'cls': monai.losses.DiceFocalLoss, 'input_style': 'prob', 'target_style': 'ohe'},
#         {'cls': monai.losses.GeneralizedDiceLoss, 'input_style': 'prob', 'target_style': 'ohe'},
#         {'cls': monai.losses.GeneralizedWassersteinDiceLoss, 'input_style': 'prob', 'target_style': 'idx', 'kwargs': {'dist_matrix': dist_matrix}},
#         # {'cls': monai.losses.DiceFocalLoss, 'input_style': 'logit', 'target_style': 'ohe'},
#         # {'cls': torch.nn.CrossEntropyLoss, 'input_style': 'logit', 'target_style': 'idx'},
#         {'cls': CrossEntropyLoss_KeepDim, 'input_style': 'logit', 'target_style': 'idx'},
#         # torch.nn.NLLLoss,
#         # torch.nn.BCELoss,
#         {'cls': torch.nn.BCEWithLogitsLoss, 'input_style': 'logit', 'target_style': 'ohe'},
#     ]

#     for loss_info in loss_infos:
#         kwargs = loss_info.get('kwargs', {})
#         loss_info['instance'] = loss_info['cls'](reduction='none', **kwargs)
#         loss_info['name'] = loss_info['cls'].__name__

#     if 0:
#         # Create structured "Truth" maps
#         true_cxs = torch.zeros(B, H, W).long()
#         true_cxs[:] = (torch.arange(B) % C)[:, None, None]
#         true_ohe = kwarray.one_hot_embedding(true_cxs, num_classes=C, dim=1)

#         # Create structured "Prediction" maps
#         prob = torch.full((B, C, H, W), fill_value=0.5)

#         prob_for_real_positive = torch.linspace(eps, 1 - eps, B)[:, None, None, None] * true_ohe
#         total_probs_for_real_negative = 1 - prob_for_real_positive.sum(axis=1, keepdim=True)
#         other_frac = torch.rand(B, C, H, W).softmax(dim=1) * (1 - true_ohe)
#         other_frac = other_frac / other_frac.sum(dim=1, keepdim=True)
#         probs_for_real_negative = other_frac * total_probs_for_real_negative
#         prob = (prob_for_real_positive + probs_for_real_negative)
#         assert torch.allclose(prob.sum(dim=1), torch.Tensor([1.0]))
#     else:
#         # Setup imbalanced classes
#         thresh = torch.linspace(0, 1, C + 1)[0:C] ** 0.5
#         true_cxs = torch.zeros(B * H * W)
#         chance = torch.rand(true_cxs.shape)
#         true_cxs = (torch.searchsorted(thresh, chance) - 1).clamp(0, C)
#         true_cxs = true_cxs.view(B, H, W)
#         true_ohe = kwarray.one_hot_embedding(true_cxs, num_classes=C, dim=1)

#         flat_ohe = einops.rearrange(true_ohe, 'b c h w -> (b h w) c')
#         flat_pos_prob = np.linspace(0, 1, len(flat_ohe))
#         rng = np.random.RandomState(0)
#         rng.shuffle(flat_pos_prob)

#         # Populate negative probs
#         flat_neg_prob = torch.rand(flat_ohe.shape).softmax(dim=1) * (1 - flat_ohe)
#         norm_factor = (1 - flat_pos_prob[:, None]) / flat_neg_prob.sum(dim=1, keepdim=True)
#         flat_neg_prob = flat_neg_prob * norm_factor
#         # Populate positive probs
#         flat_prob = (flat_neg_prob + (flat_ohe * flat_pos_prob[:, None])).float()
#         assert torch.allclose(flat_prob.sum(dim=1), torch.Tensor([1.0]))

#         prob = einops.rearrange(flat_prob, '(b h w) c -> b c h w', b=B, h=H, w=W, c=C)

#     # Different loss criterions require different encodings of the truth
#     targets = {
#         'ohe': true_ohe,  # one hot embedding
#         'idx': true_cxs,  # true class index
#     }

#     # Invert probabilities into raw network "logit" outputs
#     inputs = {
#         'logit': inverse_sigmoid(prob),
#         'prob': prob,
#     }

#     # positive_class_idx = (C - 1)
#     positive_class_idx = 1

#     rows = []

#     # import kwarray
#     _impl = kwarray.ArrayAPI.coerce('torch')

#     for loss_info in loss_infos:

#         input = inputs[loss_info['input_style']]
#         target = targets[loss_info['target_style']]
#         loss_instance = loss_info['instance']

#         loss_values = loss_instance(input, target)
#         ohe = targets['ohe']
#         prob = inputs['prob']

#         print(loss_info['name'])
#         print('loss_values.shape = {!r}'.format(loss_values.shape))

#         if H == 1 and W == 1:
#             if tuple(loss_values.shape) == (B, C):
#                 loss_values = loss_values.view(B, C, 1, 1)
#             elif tuple(loss_values.shape) == (B,):
#                 loss_values = loss_values.view(B, 1, 1, 1)

#         if tuple(loss_values.shape) == (B, C, H, W):
#             flat_prob = einops.rearrange(prob, 'b c h w -> (b h w) c')
#             flat_loss = einops.rearrange(loss_values, 'b c h w -> (b h w) c')
#             flat_ohe = einops.rearrange(ohe, 'b c h w -> (b h w) c')
#             is_pos = flat_ohe[:, positive_class_idx].bool()
#             true_pos_prob = flat_prob[:, positive_class_idx][is_pos]
#             true_pos_loss = flat_loss[:, positive_class_idx][is_pos]

#             false_pos_prob_stack = []
#             false_pos_loss_stack = []
#             for neg_cx in set(range(C)) - {positive_class_idx}:
#                 is_neg = flat_ohe[:, neg_cx].bool()
#                 _false_pos_prob = flat_prob[:, positive_class_idx][is_neg]
#                 _false_pos_loss = flat_loss[:, positive_class_idx][is_neg]
#                 false_pos_prob_stack.append(_false_pos_prob)
#                 false_pos_loss_stack.append(_false_pos_loss)
#             false_pos_prob = torch.cat(false_pos_prob_stack)
#             false_pos_loss = torch.cat(false_pos_loss_stack)

#         elif tuple(loss_values.shape) == (B, 1, H, W):
#             flat_prob = einops.rearrange(prob, 'b c h w -> (b h w) c')
#             flat_loss = einops.rearrange(loss_values, 'b c h w -> (b h w) c')
#             flat_ohe = einops.rearrange(ohe, 'b c h w -> (b h w) c')
#             is_pos = flat_ohe[:, positive_class_idx].bool()
#             true_pos_prob = flat_prob[:, positive_class_idx][is_pos]
#             true_pos_loss = flat_loss[:, 0][is_pos]
#         else:
#             continue

#         rows += [
#             {
#                 'pos_prob': p,
#                 'loss': ell,
#                 'type': loss_info['name'],
#                 'case': 'pos',
#                 'type2': loss_info['name'] + '_tp',
#             }
#             for p, ell in zip(_impl.tolist(true_pos_prob), _impl.tolist(true_pos_loss))
#         ]
#         rows += [
#             {
#                 'pos_prob': p,
#                 'loss': ell,
#                 'type': loss_info['name'],
#                 'case': 'neg',
#                 'type2': loss_info['name'] + '_fp',
#             }
#             for p, ell in zip(_impl.tolist(false_pos_prob), _impl.tolist(false_pos_loss))
#         ]

#     import kwplot
#     import pandas as pd
#     sns = kwplot.autosns()
#     df = pd.DataFrame(rows)
#     df = df[np.isfinite(df.loss)]
#     print(df)
#     print('df.shape = {!r}'.format(df.shape))

#     groups = dict(list(df.groupby('type')))

#     min_loss = 0
#     max_loss = df.loss.max()

#     pnum_ = kwplot.PlotNums(nSubplots=len(groups))
#     kwplot.figure(fnum=1, doclf=True)
#     for group_name, group in groups.items():
#         kwplot.figure(fnum=1, pnum=pnum_())
#         ax = sns.lineplot(data=group, x='pos_prob', y='loss', hue='case')
#         ax.set_yscale('symlog')
#         ax.set_title(group_name)
#         ax.set_ylim(min_loss, max_loss)


# def nll_focal_loss(log_probs, targets, focus, dim=1, weight=None,
#                    ignore_index=None, reduction='none'):
#     r"""
#     Focal loss given preprocessed log_probs (log probs) instead of raw outputs

#     Args:
#         log_probs (FloatTensor): log-probabilities for each class
#         targets (LongTensor): correct class indices for each example
#         focus (float): focus factor
#         dim (int, default=1): class dimension (usually 1)
#         weight (FloatTensor): per-class weights
#         ignore_index (int, default=None):
#         reduction (str):
#     """
#     import kwarray
#     if focus == 0 and dim == 1:
#         # In this case nll_focal_loss is nll_loss, but nll_loss is faster
#         if ignore_index is None:
#             ignore_index = -100
#         return F.nll_loss(log_probs, targets, weight=weight,
#                           ignore_index=ignore_index, reduction=reduction)

#     # Determine which entry in log_probs corresponds to the target
#     num_classes = log_probs.shape[dim]
#     t = kwarray.one_hot_embedding(targets.data, num_classes, dim=dim)

#     # We only need the log(p) component corresponding to the target class
#     target_log_probs = (log_probs * t).sum(dim=dim)  # sameas log_probs[t > 0]

#     # Modulate the weight of examples based on hardness
#     target_p = torch.exp(target_log_probs)
#     w = (1 - target_p).pow(focus)

#     # Factor in per-class `weight` to the a per-input weight
#     if weight is not None:
#         class_weight = weight[targets]
#         w *= class_weight

#     if ignore_index is not None:
#         # remove any loss associated with ignore_label
#         ignore_mask = (targets != ignore_index).float()
#         w *= ignore_mask

#     # Normal cross-entropy computation (but with augmented weights per example)
#     # Recall the nll_loss of an aexample is simply its -log probability or the
#     # real class, all other classes are not needed (due to softmax magic)
#     output = w * -target_log_probs

#     if reduction ==  'mean':
#         output = output.mean()
#     elif reduction == 'sum':
#         output = output.sum()
#     elif reduction == 'none':
#         pass
#     else:
#         raise KeyError(reduction)
#     return output


# class NetharnFocalLoss(torch.nn.modules.loss._WeightedLoss):
#     r"""
#     Generalization of ``CrossEntropyLoss`` with a "focus" modulation term.

#     Original implementation in [1]_.

#     .. math::
#         FL(p_t) = - \alpha_t * (1 − p[t]) ** γ * log(p[t]).
#         focal_loss(x, class) = weight[class] * (-x[class] + log(\sum_j exp(x[j])))

#     PythonMath:
#         FL(p[t]) = -α[t] * (1 − p[t]) ** γ * log(p[t]).

#     Args:
#         focus (float): Focusing parameter. Equivelant to Cross Entropy when
#             `focus == 0`. (Defaults to 2) (Note: this is gamma in the paper)

#         weight (Tensor, optional): a manual rescaling weight given to each
#            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
#            treated as if having all ones.


#            Finally we note that α, the weight assigned to the rare class, also
#            has a stable range, but it interacts with γ making it necessary to
#            select the two together

#            This should be set depending on `focus`. See [2] for details.
#            In general α should be decreased slightly as γ is increased
#            (Note: this is α in the paper)

#            α ∈ [0, 1] for class 1 and 1−α for class −1

#         size_average (bool, optional): By default, the losses are averaged
#            over observations for each minibatch. However, if the field
#            size_average is set to ``False``, the losses are instead summed for
#            each minibatch. Ignored when reduce is ``False``. Default: ``True``

#         reduce (bool, optional): By default, the losses are averaged or summed
#            for each minibatch. When reduce is ``False``, the loss function returns
#            a loss per batch element instead and ignores size_average.
#            Default: ``True``

#         ignore_index (int, optional): Specifies a target value that is ignored
#             and does not contribute to the input gradient. When size_average is
#             ``True``, the loss is averaged over non-ignored targets.

#     References:
#         [1] https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
#         [2] https://arxiv.org/abs/1708.02002

#     SeeAlso:
#         https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
#         https://discuss.pytorch.org/t/how-to-implement-focal-loss-in-pytorch/6469/11
#     """

#     def __init__(self, focus=2, weight=None, reduction='mean',
#                  ignore_index=-100):
#         if isinstance(weight, list):
#             weight = torch.FloatTensor(weight)
#         super().__init__(weight=weight, reduction=reduction)
#         self.focus = focus
#         self.ignore_index = ignore_index

#     def forward(self, input, target):
#         """
#         Args:
#           input: (tensor) predicted class confidences, sized [batch_size, #classes].
#           target: (tensor) encoded target labels, sized [batch_size].

#         Returns:
#             (tensor) loss

#         CommandLine:
#             python -m netharn.loss FocalLoss.forward:0 --profile
#             python -m netharn.loss FocalLoss.forward:1 --profile

#         CommandLine:
#             xdoctest -m netharn.criterions.focal FocalLoss.forward

#         Example:
#             >>> from netharn.criterions.focal import *  # NOQA
#             >>> import numpy as np
#             >>> # input is of size B x C
#             >>> B, C = 8, 5
#             >>> # each element in target has to have 0 <= value < C
#             >>> target = (torch.rand(B) * C).long()
#             >>> input = torch.randn(B, C, requires_grad=True)
#             >>> # Check to be sure that when gamma=0, FL becomes CE
#             >>> loss0 = FocalLoss(reduction='none', focus=0).forward(input, target)
#             >>> loss1 = F.cross_entropy(input, target, reduction='none')
#             >>> #loss1 = F.cross_entropy(input, target, size_average=False, reduce=False)
#             >>> loss2 = F.nll_loss(F.log_softmax(input, dim=1), target, reduction='none')
#             >>> #loss2 = F.nll_loss(F.log_softmax(input, dim=1), target, size_average=False, reduce=False)
#             >>> assert np.all(np.abs((loss1 - loss0).data.numpy()) < 1e-6)
#             >>> assert np.all(np.abs((loss2 - loss0).data.numpy()) < 1e-6)
#             >>> lossF = FocalLoss(reduction='none', focus=2, ignore_index=0).forward(input, target)
#             >>> weight = torch.rand(C)
#             >>> lossF = FocalLoss(reduction='none', focus=2, weight=weight, ignore_index=0).forward(input, target)

#         Ignore:
#             >>> from netharn.criterions.focal import *  # NOQA
#             >>> import numpy as np
#             >>> B, C = 8, 5
#             >>> target = (torch.rand(B) * C).long()
#             >>> input = torch.randn(B, C, requires_grad=True)
#             >>> for reduction in ['sum', 'none', 'mean']:
#             >>>     fl0 = FocalLoss(reduction=reduction, focus=0)
#             >>>     fl2 = FocalLoss(reduction=reduction, focus=2)
#             >>>     cce = torch.nn.CrossEntropyLoss(reduction=reduction)
#             >>>     output1 = fl0(input, target).data.numpy()
#             >>>     output2 = fl2(input, target).data.numpy()
#             >>>     output3 = cce(input, target).data.numpy()
#             >>>     assert np.all(np.isclose(output1, output3))
#         """
#         nll = input.log_softmax(dim=1)
#         return nll_focal_loss(
#             nll, target, focus=self.focus, dim=1, weight=self.weight,
#             ignore_index=self.ignore_index, reduction=self.reduction)
