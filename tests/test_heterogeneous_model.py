
def test_heterogeneous_with_split_attention_backbone():
    """
    Setup a test dataset, make a heterogenous model and a multimodal model and
    check weights transfer between the two
    """
    from geowatch.tasks.fusion.methods.heterogeneous import HeterogeneousModel
    from geowatch.tasks.fusion.methods.heterogeneous import ScaleAgnostictPositionalEncoder
    from geowatch.tasks.fusion.methods.channelwise_transformer import MultimodalTransformer
    import ubelt as ub

    from geowatch.tasks.fusion import datamodules
    print('(STEP 0): SETUP THE DATA MODULE')
    datamodule = datamodules.KWCocoVideoDataModule(
        train_dataset='special:vidshapes-geowatch', num_workers=0, channels='auto')
    datamodule.setup('fit')
    dataset = datamodule.torch_datasets['train']
    print('(STEP 1): ESTIMATE DATASET STATS')
    dataset_stats = dataset.cached_dataset_stats(num=3)
    print('dataset_stats = {}'.format(ub.urepr(dataset_stats, nl=3)))
    loader = datamodule.train_dataloader()
    print('(STEP 2): SAMPLE BATCH')
    batch = next(iter(loader))
    for item_idx, item in enumerate(batch):
        print(f'item_idx={item_idx}')
        for frame_idx, frame in enumerate(item['frames']):
            print(f'  * frame_idx={frame_idx}')
            print(f'  * frame.sensor = {frame["sensor"]}')
            for mode_code, mode_val in frame['modes'].items():
                print(f'      * {mode_code=} @shape={mode_val.shape}, num_nam={mode_val.isnan().sum()}')
    print('(STEP 3): THE REST OF THE TEST')

    position_encoder = ScaleAgnostictPositionalEncoder(3, 8)

    model1 = HeterogeneousModel(
        dataset_stats=dataset_stats,
        classes=datamodule.classes,
        position_encoder=position_encoder,
        backbone='smt_it_joint_p2',
    )

    model2 = MultimodalTransformer(
        arch_name='smt_it_joint_p2',
        dataset_stats=dataset_stats,
        classes=datamodule.classes,
        change_loss='dicefocal')

    # Test that weights transfer
    import torch_liberator
    stats = torch_liberator.load_partial_state(model1, model2.state_dict())
    mapping = stats['mapping']
    assert sum('encoder.layers.0' in k for k in mapping) > 4
    assert sum('encoder.layers.1' in k for k in mapping) > 4

    # Test that batches are accepted
    # Note: this doesn't work yet.
    if 0:
        outputs1 = model1.forward(batch)
    else:
        import warnings
        warnings.warn('warning: split attention heterogeneous model is not implemented')
    outputs2 = model2.forward_step(batch)
