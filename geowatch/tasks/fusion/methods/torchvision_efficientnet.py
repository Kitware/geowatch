import pytorch_lightning as pl


class ConvNetWrapper(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.ots_model = self.define_ots_model()

    def define_ots_model(self):
        raise NotImplementedError('Child class must define this')


class TorchvisionEfficientDet(ConvNetWrapper):
    def __init__(self):
        super().__init__()

    def define_ots_model(self):
        import torchvision
        ots_model = torchvision.models.efficientnet_b7(weights=True)
        return ots_model
