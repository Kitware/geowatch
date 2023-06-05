import segmentation_models_pytorch as smp


def create_smp_network(model_name,
                       encoder_name,
                       in_channels,
                       out_channels,
                       pretrain=None,
                       **kwargs):
    network = smp.create_model(model_name,
                               encoder_name=encoder_name,
                               in_channels=in_channels,
                               encoder_weights=pretrain,
                               classes=out_channels,
                               **kwargs)
    return network


if __name__ == '__main__':
    model_name = 'unetplusplus'
    encoder_name = 'resnet50'
    in_channels = 4
    out_channels = 8
    pretrain = 'imagenet'
    model = create_smp_network(model_name, encoder_name, in_channels, out_channels, pretrain)
