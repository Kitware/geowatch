from geowatch.monkey import monkey_lightning
from geowatch.monkey import monkey_tensorflow
from geowatch.monkey import monkey_numpy
monkey_tensorflow.disable_tensorflow_warnings()
monkey_lightning.disable_lightning_hardware_warnings()
monkey_numpy.patch_numpy_2x()


def test_save_channelwise_plain():
    # Use one of our fusion.architectures in a test
    from geowatch.tasks.fusion import methods
    model = methods.MultimodalTransformer(
        arch_name="smt_it_joint_p2", input_sensorchan=5,
        change_head_hidden=0, saliency_head_hidden=0,
        class_head_hidden=0)
    _save_package(model)


def test_save_channelwise_with_dataloader():
    from geowatch.tasks.fusion import datamodules
    from geowatch.tasks.fusion import methods
    import kwcoco
    dset = kwcoco.CocoDataset.demo('special:vidshapes8-multispectral-multisensor', rng=0)
    datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
        train_dataset=dset.fpath,
        batch_size=1, time_steps=2, num_workers=2, normalize_inputs=10,
        channels='auto')
    datamodule.setup('fit')
    dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=30)

    classes = datamodule.torch_datasets['train'].predictable_classes

    # Use one of our fusion.architectures in a test
    model = methods.MultimodalTransformer(
        arch_name="smt_it_joint_p2", classes=classes,
        dataset_stats=dataset_stats, input_sensorchan=datamodule.input_sensorchan,
        learning_rate=1e-8, optimizer='sgd',
        change_head_hidden=0, saliency_head_hidden=0,
        class_head_hidden=0)
    _save_package_with_trainer(model, datamodule)


def test_save_heterogeneous_plain():
    # Use one of our fusion.architectures in a test
    from geowatch.tasks.fusion import methods
    from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
    position_encoder = methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
    backbone = TransformerEncoderDecoder(
        encoder_depth=1,
        decoder_depth=1,
        dim=position_encoder.output_dim + 16,
        queries_dim=position_encoder.output_dim,
        logits_dim=16,
        cross_heads=1,
        latent_heads=1,
        cross_dim_head=1,
        latent_dim_head=1,
    )
    model = methods.HeterogeneousModel(
        position_encoder=position_encoder,
        input_sensorchan=5,
        decoder="upsample",
        backbone=backbone,
    )
    _save_package(model)


def test_save_heterogeneous_with_dataloader():
    # Use one of our fusVon.architectures in a test
    from geowatch.tasks.fusion import methods
    from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
    position_encoder = methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
    backbone = TransformerEncoderDecoder(
        encoder_depth=1,
        decoder_depth=1,
        dim=position_encoder.output_dim + 16,
        queries_dim=position_encoder.output_dim,
        logits_dim=16,
        cross_heads=1,
        latent_heads=1,
        cross_dim_head=1,
        latent_dim_head=1,
    )
    model = methods.HeterogeneousModel(
        position_encoder=position_encoder,
        input_sensorchan='*:r|g|b',
        decoder="upsample",
        backbone=backbone,
    )
    from geowatch.tasks.fusion import datamodules
    datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
        train_dataset='special:vidshapes1', window_space_dims=32,
        batch_size=1, time_steps=2, num_workers=2, normalize_inputs=10, channels='auto')
    datamodule.setup('fit')
    _save_package_with_trainer(model, datamodule)


def test_save_unet_plain():
    # Use one of our fusion.architectures in a test
    from geowatch.tasks.fusion import methods
    model = methods.UNetBaseline(
        input_sensorchan=5,
    )
    _save_package(model)


def test_save_unet_with_dataloader():
    import pytest
    pytest.skip('not working')
    # Use one of our fusion.architectures in a test
    from geowatch.tasks.fusion import methods
    model = methods.UNetBaseline(
        input_sensorchan='*:r|g|b',
    )
    from geowatch.tasks.fusion import datamodules
    datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
        train_dataset='special:vidshapes8', chip_size=32,
        batch_size=1, time_steps=2, num_workers=2, normalize_inputs=10, channels='auto')
    datamodule.setup('fit')
    _save_package_with_trainer(model, datamodule)


def test_save_noop_plain():
    # Use one of our fusion.architectures in a test
    from geowatch.tasks.fusion import methods
    model = methods.NoopModel(
        input_sensorchan=5,)
    _save_package(model)


def test_save_noop_with_dataloader():
    # Use one of our fusion.architectures in a test
    from geowatch.tasks.fusion import methods
    from geowatch.tasks.fusion import datamodules
    model = methods.NoopModel(
        input_sensorchan=5,)
    datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
        train_dataset='special:vidshapes8-multispectral-multisensor', chip_size=32,
        batch_size=1, time_steps=2, num_workers=0, normalize_inputs=10, channels='auto')
    datamodule.setup('fit')
    _save_package_with_trainer(model, datamodule)


def _save_package_with_trainer(model, datamodule):
    # Test with datamodule / trainer
    import ubelt as ub
    import pytorch_lightning as pl

    # We have to run an input through the module because it is lazy
    batch = ub.peek(iter(datamodule.train_dataloader()))
    model.training_step(batch)

    trainer = pl.Trainer(max_steps=0, accelerator='cpu', devices=1)
    trainer.fit(model=model, datamodule=datamodule)
    _save_package(model)


def _save_package(model):
    """
    Check that all of our models can be saved / reloaded via torch package.
    """
    # Test without datamodule
    import ubelt as ub
    from os.path import join
    #from geowatch.tasks.fusion.methods.heterogeneous import *  # NOQA
    name = model.__class__.__name__
    pkgid = id(model)
    dpath = ub.Path.appdir(f'geowatch/tests/package/{name}').ensuredir()
    package_path = join(dpath, f'my_package_{pkgid}.pt')

    # Save the model (TODO: need to save datamodule as well)
    model.save_package(package_path)

    # Test that the package can be reloaded
    #recon = methods.HeterogeneousModel.load_package(package_path)
    from geowatch.tasks.fusion.utils import load_model_from_package
    recon = load_model_from_package(package_path)
    # Check consistency and data is actually different
    recon_state = recon.state_dict()
    model_state = model.state_dict()
    assert recon is not model
    assert set(recon_state) == set(recon_state)
    from geowatch.utils.util_kwarray import torch_array_equal
    for key in recon_state.keys():
        assert torch_array_equal(model_state[key], recon_state[key], equal_nan=True)
        assert model_state[key] is not recon_state[key]
