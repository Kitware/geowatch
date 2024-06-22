import torch


def coerce_criterion(loss_code, weights, ohem_ratio=None, focal_gamma=2.0):
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

    Raises:
        KeyError: if loss_code is not recognized.

    Returns:
        torch.nn.modules.loss._Loss: The loss function.
    """
    # import monai
    if loss_code == 'cce':
        criterion = torch.nn.CrossEntropyLoss(
            weight=weights, reduction='none')
        criterion.target_encoding = 'index'
        criterion.logit_shape = '(b t h w) c'
        criterion.target_shape = '(b t h w)'

    elif loss_code == 'bce':
        criterion = torch.nn.BCELoss(
            weight=weights, reduction='none')
        criterion.target_encoding = 'onehot'
        criterion.logit_shape = '(b t h w) c'
        criterion.target_shape = '(b t h w)'

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
        criterion.logit_shape = '(b t h w) c'
        criterion.target_shape = '(b t h w)'

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
        criterion.logit_shape = 'b c h w t'
        criterion.target_shape = 'b c h w t'

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
        criterion.logit_shape = 'b c h w t'
        criterion.target_shape = 'b c h w t'
    else:
        # self.class_criterion = nn.CrossEntropyLoss()
        # self.class_criterion = nn.BCEWithLogitsLoss()
        raise NotImplementedError(loss_code)
    criterion.in_channels = len(weights)
    return criterion
