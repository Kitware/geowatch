# from models.unet_stn import  UNet
from watch.tasks.rutgers_material_seg.models import deeplabWS
from watch.tasks.rutgers_material_seg.models.shallow_seg import ShallowSeg
from watch.tasks.rutgers_material_seg.models.linear_classifier import VSNet

models = {
            'deeplabWS': deeplabWS,
            'shallow_seg': ShallowSeg,
            'vsnet': VSNet
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
