# from models.unet_stn import  UNet
from geowatch.tasks.rutgers_material_seg.models import deeplabWS
from geowatch.tasks.rutgers_material_seg.models.shallow_seg import ShallowSeg
from geowatch.tasks.rutgers_material_seg.models.linear_classifier import VSNet
from geowatch.tasks.rutgers_material_seg.models import resnet
from geowatch.tasks.rutgers_material_seg.models import resnetGNWS
from geowatch.tasks.rutgers_material_seg.models import resnet_enc
from geowatch.tasks.rutgers_material_seg.models import deeplab
from geowatch.tasks.rutgers_material_seg.models import deeplab_diff
from geowatch.tasks.rutgers_material_seg.models import resnet_classification_finetune

models = {
    'deeplabWS': deeplabWS,
    'shallow_seg': ShallowSeg,
    'vsnet': VSNet,
    'resnet': resnet,
    'resnetgnws': resnetGNWS,
    'resnet_enc': resnet_enc,
    'deeplab': deeplab,
    'resnet_class_ft': resnet_classification_finetune,
    "deeplab_diff": deeplab_diff
}


def build_model(model_name: str = "deeplabWS", backbone: str = "resnet101", **kwargs) -> object:
    """Model building module

    Parameters
    ----------
    model_name : str, optional
        which model to use, by default "deeplabWS"
    backbone : str, optional
        backbone of model, by default "resnet101"

    Returns
    -------
    model
        model object
    """
    if backbone:
        model = getattr(models[model_name], backbone)(**kwargs)
    else:
        model = models[model_name](**kwargs)
    return model
