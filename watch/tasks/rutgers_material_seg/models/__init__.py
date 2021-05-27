# from models.unet_stn import  UNet
from material_seg.models import deeplabWS

models = {
            'deeplabWS': deeplabWS,
            # 'unet': unet_stn,
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
    model = getattr(models[model_name], backbone)(**kwargs)
    return model