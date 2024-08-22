"""
Tests of tasks.fusion.predict using a dummy model.
"""
import pytorch_lightning as pl


class DummyModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.classes = ['star', 'superstar', 'eff']
        self.config_cli_yaml = {
            'data': {}
        }

    def forward_step(self, batch, with_loss=False):
        import torch
        import kwarray
        import kwimage
        import numpy as np
        outputs = {}

        num_classes = len(self.classes)

        # This shows how wonkey our current output API is.
        # We need to make it nicer.

        outputs['class_probs'] = []
        outputs['saliency_probs'] = []

        torch_impl = kwarray.ArrayAPI.coerce('torch')
        np_impl = kwarray.ArrayAPI.coerce('numpy')

        for item in batch:
            frame_preds = []
            for frame in item['frames']:
                # A KWCocoDataset frame requests the output shape it wants.
                class_dims = list(frame['class_output_dims'])
                saliency_dims = list(frame['saliency_output_dims'])

                # A KWCocoDataset frame also knows how to put
                # the requested prediction back in context.
                frame['output_space_slice']
                frame['output_image_dsize']
                frame['scale_outspace_from_vid']

                out_dims = frame['output_dims']

                # derived
                _derived = torch.stack([
                    v.nan_to_num().mean(dim=0)
                    for v in frame['modes'].values()]).mean(dim=0)
                derived = torch_impl.numpy(_derived).astype(np.float32)
                out_dsize = out_dims[::-1]
                derived = kwimage.imresize(derived, dsize=out_dsize)
                pattern = (derived > derived.mean())[:, :, None].astype(np.float32)
                _pattern = np_impl.tensor(pattern).to(_derived.device)

                # Pretend to make predictions
                frame_pred = {}
                frame_pred['saliency_probs'] = _pattern * torch.rand(saliency_dims + [2])
                frame_pred['class_probs'] =  _pattern * torch.rand(class_dims + [num_classes])
                frame_preds.append(frame_pred)

            outputs['class_probs'].append(
                torch.stack([f['class_probs'] for f in frame_preds])
            )
            outputs['saliency_probs'].append(
                torch.stack([f['saliency_probs'] for f in frame_preds])
            )
        return outputs


class BatchOutputsV1(dict):
    """
    Expected structure

    Dict:
        class_probs (List[Tensor]): A list item for each batch item containing a TxHxWxC tensor.
        saliency_probs (List[Tensor]): A list item for each batch item containing a TxHxWx2 tensor.
        loss_parts (Dict[str, Tensor]): parts of the loss function to combine
        total_loss (Scalar): the final loss to backpropogate on
    """


class ItemOutputV2(dict):
    """
    Expected Structure:
        'frames': List[Dict]: a dictionary containing predictions for each frame / different timestep containin keys
            {
                'class' (Tensor): class prediction for this frame
                'saliency' (Tensor): saliency prediction for this frame
            }
    """


class BatchOutputsV2(dict):
    """
    Expected structure:

    Dict:
        outputs (List[ItemOutputV1]): A list item for each batch item containing a dictionary with
        loss_parts (Dict[str, Tensor]): parts of the loss function to combine
        total_loss (Scalar): the final loss to backpropogate on
    """


def ensure_test_kwcoco_dataset():
    import geowatch
    import ubelt as ub
    import kwcoco
    dpath = ub.Path.appdir('geowatch/tests/fusion_predict/demodata').ensuredir()
    test_dataset_fpath = dpath / 'test.kwcoco.zip'
    # Create a dataset with multiple different image sizes
    if not test_dataset_fpath.exists():
        dset1 = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True, num_frames=16, image_size=(64, 64), num_videos=2)
        dset1.clear_annotations()
        dset2 = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True, num_frames=16, image_size=(128, 128), num_videos=2)
        dset2.clear_annotations()
        dset1.reroot(absolute=True)
        dset2.reroot(absolute=True)
        dset = kwcoco.CocoDataset.union(dset1, dset2)
        dset.fpath = test_dataset_fpath
        dset.dump()
    return test_dataset_fpath


def test_predict_with_dummy_model():
    import ubelt as ub
    from geowatch.tasks.fusion import predict as predict_mod
    from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
    disable_lightning_hardware_warnings()

    dpath = ub.Path.appdir('geowatch/tests/fusion_predict/dummy_model').ensuredir()

    pred_dataset_fpath = dpath / 'pred.kwcoco.zip'
    test_dataset_fpath = dpath / 'test.kwcoco.zip'

    test_dataset_fpath = ensure_test_kwcoco_dataset()

    model = DummyModel()

    # Predict via that model
    kwargs = {
        'model': model,
        'pred_dataset': pred_dataset_fpath,
        'test_dataset': test_dataset_fpath,
        'channels': 'r|g|b',
        'batch_size': 1,
        'num_workers': 0,
        'devices': 'cpu',
    }
    config = predict_mod.PredictConfig(**kwargs)
    predictor = predict_mod.Predictor(config)

    predictor._load_model()
    predictor._load_dataset()

    # Execute the pipeline
    result_dataset = predictor._run_critical_loop()

    if 0:
        import ubelt as ub
        ub.cmd(f'geowatch stats {result_dataset.fpath}', verbose=3, system=True)
        ub.cmd(f'geowatch visualize {result_dataset.fpath} --channels="r|g|b,star|superstar|eff,salient" --stack=True', verbose=3, system=True)


def test_predict_with_dummy_model_memmap():
    import ubelt as ub
    from geowatch.tasks.fusion import predict as predict_mod
    from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
    disable_lightning_hardware_warnings()

    dpath = ub.Path.appdir('geowatch/tests/fusion_predict/memmap').ensuredir()

    pred_dataset_fpath = dpath / 'pred.kwcoco.zip'
    test_dataset_fpath = ensure_test_kwcoco_dataset()

    model = DummyModel()

    # Predict via that model
    kwargs = {
        'model': model,
        'pred_dataset': pred_dataset_fpath,
        'test_dataset': test_dataset_fpath,
        'channels': 'r|g|b',
        'batch_size': 1,
        'num_workers': 0,
        'devices': 'cpu',
        'memmap': True,
    }
    config = predict_mod.PredictConfig(**kwargs)
    predictor = predict_mod.Predictor(config)

    predictor._load_model()
    predictor._load_dataset()

    # Execute the pipeline
    result_dataset = predictor._run_critical_loop()

    if 0:
        import ubelt as ub
        ub.cmd(f'geowatch stats {result_dataset.fpath}', verbose=3, system=True)
        ub.cmd(f'geowatch visualize {result_dataset.fpath} --channels="r|g|b,star|superstar|eff,salient" --stack=True', verbose=3, system=True)


def test_predict_with_dummy_model_lower_resolution():
    """
    pytest -s -k test_predict_with_dummy_model_lower_resolution
    """
    import ubelt as ub
    from geowatch.tasks.fusion import predict as predict_mod
    from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
    disable_lightning_hardware_warnings()

    dpath = ub.Path.appdir('geowatch/tests/fusion_predict/lower_resolution').ensuredir()

    pred_dataset_fpath = dpath / 'pred.kwcoco.zip'

    test_dataset_fpath = ensure_test_kwcoco_dataset()

    model = DummyModel()

    # Predict via that model
    kwargs = {
        'model': model,
        'pred_dataset': pred_dataset_fpath,
        'test_dataset': test_dataset_fpath,
        'channels': 'r|g|b',
        'batch_size': 1,
        'num_workers': 0,
        'devices': 'cpu',

        'input_space_scale': '0.2 GSD',
        'output_space_scale': '1.0 GSD',
    }
    config = predict_mod.PredictConfig(**kwargs)
    predictor = predict_mod.Predictor(config)

    predictor._load_model()
    predictor._load_dataset()

    dataset = predictor.datamodule.torch_datasets['test']
    # Look at one batch
    item = dataset[0]
    batch = [item]

    # Smoke test before we execute prediction
    outputs = model.forward_step(batch)
    print('item_summary ' + ub.urepr(dataset.summarize_item(item), nl=2))
    outputs['class_probs'][0].shape
    outputs['saliency_probs'][0].shape

    # Execute the pipeline
    result_dataset = predictor._run_critical_loop()

    if 0:
        import ubelt as ub
        ub.cmd(f'geowatch stats {result_dataset.fpath}', verbose=3, system=True)
        ub.cmd(f'geowatch visualize {result_dataset.fpath} --channels="r|g|b,star|superstar|eff,salient" --stack=True', verbose=3, system=True)
