import torch
import numpy as np

from segmentation_models_pytorch.losses import JaccardLoss

from geowatch.tasks.rutgers_material_seg_v2.matseg.models.pl_wrapper import MaterialSegmentationModel
from geowatch.tasks.rutgers_material_seg_v2.matseg.models.loss import CrossEntropyLossWithOHEM


def build_model(class_count,
                network_name,
                encoder_name,
                in_channels,
                out_channels,
                loss_mode,
                optimizer_mode,
                class_weight_mode,
                lr,
                wd,
                lr_scheduler_mode,
                pretrain=None,
                to_rgb_fcn=None,
                checkpoint_path=None,
                ohem_ratio=0.75):

    # Get class weights.
    if class_count is None:
        class_weights = None
    else:
        class_weights = get_class_weights(class_count, out_channels, class_weight_mode)

    # Get model parameters.
    model_params = {
        'network_name': network_name,
        'encoder_name': encoder_name,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'pretrain': pretrain
    }

    # Build loss function.
    loss_func = get_loss_funtion(loss_mode, class_weights, ignore_index=0, ohem_ratio=ohem_ratio)

    # Put into the PL wrapper.
    model = MaterialSegmentationModel(model_params,
                                      loss_func,
                                      out_channels,
                                      lr=lr,
                                      wd=wd,
                                      optimizer_mode=optimizer_mode,
                                      lr_scheduler_mode=lr_scheduler_mode)

    if checkpoint_path is not None:
        print(f'Loading model weights from: {checkpoint_path}')
        model = model.load_from_checkpoint(checkpoint_path)

    return model


def get_loss_funtion(loss_mode, class_weights, ignore_index=-100, ohem_ratio=0.75):
    """Get objective function for model training.

    Args:
        loss_mode (str): The loss function to use.
        class_weights (torch.Tensor(float)): The weight factor for each class.
            If None, all classes are equally weighted.
        ignore_index (int, optional): The class index to ignore. Defaults to -100.
        ohem_ratio (float, optional): The percentage [0,1] of values to keep for
            computing the loss. As defined by original paper:
            https://arxiv.org/pdf/1604.03540.pdf. Defaults to 0.75.

    Raises:
        NotImplementedError: Loss mode not implemented.

    Returns:
        torch.nn.Module: The loss function with a forward method.
    """
    if loss_mode == 'cross_entropy':
        loss_func = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    elif loss_mode == 'jaccard':
        if class_weights is not None:
            print('WARNING: Jaccard loss does not support class weights.')
        loss_func = JaccardLoss(mode='multiclass')
    elif loss_mode == 'cross_entropy_ohem':
        loss_func = CrossEntropyLossWithOHEM(ohem_ratio=ohem_ratio,
                                             weight=class_weights,
                                             ignore_index=ignore_index)
    else:
        raise NotImplementedError(f'Loss mode {loss_mode} not implemented.')
    return loss_func


def get_class_weights(class_count, out_channels, class_weight_mode):
    if class_count is None:
        class_weights = np.ones(out_channels, dtype='float32')
    else:
        if class_weight_mode == 'equal':
            class_weights = np.ones(out_channels, dtype='float32')
        elif class_weight_mode == 'prop':
            if isinstance(class_count, dict):
                n_annos = sum(list(class_count.values()))
                class_weights = np.zeros(out_channels, dtype='float32')
                for mat_id, n_class_annos in class_count.items():
                    class_weights[mat_id] = n_annos / (n_class_annos * out_channels)
            elif isinstance(class_count, np.ndarray):
                n_annos = class_count.sum()
                class_weights = np.zeros(out_channels, dtype='float32')
                for i in range(class_count.shape[0]):
                    class_weights[i] = n_annos / (class_count[i] * out_channels)
            else:
                raise NotImplementedError

        elif class_weight_mode == 'sqrt':
            if isinstance(class_count, dict):
                n_annos = sum(list(class_count.values()))
                class_weights = np.zeros(out_channels, dtype='float32')
                for mat_id, n_class_annos in class_count.items():
                    class_weights[mat_id] = np.sqrt(n_annos / (n_class_annos * out_channels))
            elif isinstance(class_count, np.ndarray):
                n_annos = class_count.sum()
                class_weights = np.zeros(out_channels, dtype='float32')
                for i in range(class_count.shape[0]):
                    class_weights[i] = np.sqrt(n_annos / (class_count[i] * out_channels))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    # pylint: disable-next=Pylint(E1101:no-member)
    class_weights = torch.from_numpy(class_weights)

    return class_weights
