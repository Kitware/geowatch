import torch


def coerce_criterion(loss_code, weights, ohem_ratio=None, focal_gamma=2.0,
                     spatial_dims='legacy'):
    """
    Helps build a loss function and returns information about the shapes needed
    by the specific loss. Augments the criterion with extra information about
    what it expects.

    Args:
        loss_code (str): The code that corresponds to loss function call.
            One of ['cce', 'focal', 'dicefocal'].
        weights (torch.Tensor): Per class weights.
            Note: Only used for 'cce' and 'focal' losses.
        ohem_ratio (float): Ratio of hard examples to sample to compute loss.
            Note: Only applies to focal losses.
        focal_gamma (float): Focal loss gamma parameter.

        spatial_dims (str):
            A code indicating which spatial dimension we are expecting in this
            loss. The "legacy" maintains backwards compat with the multimodal
            transformer. For spacetime segmentation this should usually be
            't h w'. For nonlocal it should be ''.

    Raises:
        KeyError: if loss_code is not recognized.

    Returns:
        torch.nn.modules.loss._Loss: The loss function.

        The loss criterion will contain variables:

            target_encoding: which is either index or onehot
            logit_shape: the expected shape of the predicted logits.
            target_shape: the expected shape of the truth targets.
    """
    # import monai
    if loss_code == 'cce':
        criterion = torch.nn.CrossEntropyLoss(
            weight=weights, reduction='none')
        criterion.target_encoding = 'index'
        if spatial_dims == 'legacy':
            spatial_dims = 't h w'
        criterion.logit_shape = f'(b {spatial_dims}) c'
        criterion.target_shape = f'(b {spatial_dims})'

    elif loss_code == 'bce':
        criterion = torch.nn.BCELoss(
            weight=weights, reduction='none')
        criterion.target_encoding = 'onehot'
        if spatial_dims == 'legacy':
            spatial_dims = 't h w'
        criterion.logit_shape = '(b {spatial_dims}) c'
        criterion.target_shape = f'(b {spatial_dims}) c'

    elif loss_code == 'focal_multiclass':
        from geowatch.utils.ext_monai import FocalLoss
        # from monai.losses import FocalLoss
        criterion = FocalLoss(
            reduction='none',
            to_onehot_y=True,
            weight=weights,
            ohem_ratio=ohem_ratio,
            gamma=focal_gamma)

        criterion.target_encoding = 'index'
        if spatial_dims == 'legacy':
            spatial_dims = 't h w'
        criterion.logit_shape = f'(b {spatial_dims}) c'
        criterion.target_shape = f'(b {spatial_dims})'

    elif loss_code == 'focal':
        from geowatch.utils.ext_monai import FocalLoss
        # from monai.losses import FocalLoss
        criterion = FocalLoss(
            reduction='none',
            to_onehot_y=False,
            weight=weights,
            ohem_ratio=ohem_ratio,
            gamma=focal_gamma)
        criterion.target_encoding = 'onehot'
        if spatial_dims == 'legacy':
            spatial_dims = 'h w t'
        criterion.logit_shape = f'b c {spatial_dims}'
        criterion.target_shape = f'b c {spatial_dims}'

    elif loss_code == 'dicefocal':
        from geowatch.utils.ext_monai import DiceFocalLoss
        # from monai.losses import DiceFocalLoss
        criterion = DiceFocalLoss(
            sigmoid=True,
            to_onehot_y=False,
            focal_weight=weights,
            reduction='none',
            ohem_ratio_focal=ohem_ratio,
            gamma=focal_gamma)
        criterion.target_encoding = 'onehot'
        if spatial_dims == 'legacy':
            spatial_dims = 'h w t'
        criterion.logit_shape = f'b c {spatial_dims}'
        criterion.target_shape = f'b c {spatial_dims}'
    else:
        # self.class_criterion = nn.CrossEntropyLoss()
        # self.class_criterion = nn.BCEWithLogitsLoss()
        raise NotImplementedError(loss_code)

    criterion.in_channels = len(weights)
    return criterion
