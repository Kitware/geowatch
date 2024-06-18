"""
Originally these were doctests in the dataloader, but were removed for space.

TODO: These should be refactored into more descriptive tests that target
specific parameters in the datamodule config
"""
from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
import ndsampler
import kwcoco
import geowatch


def test_watch_tasks_fusion_datamodules_kwcoco_dataset_full_KWCocoVideoDataset():
    """
    converted from /home/joncrall/code/watch/geowatch/tasks/fusion/datamodules/_orig_kwcoco_dataset_full.py::KWCocoVideoDataset:1
    """
    coco_dset = geowatch.coerce_kwcoco('geowatch')
    print({c.get('sensor_coarse') for c in coco_dset.images().coco_images})
    print({c.channels.spec for c in coco_dset.images().coco_images})
    sampler = ndsampler.CocoSampler(coco_dset)
    self = KWCocoVideoDataset(sampler, time_dims=2, window_dims=(128, 128),
                              channels=None)
    index = 0
    item = self[index]
    canvas = self.draw_item(item)
    canvas


def test_watch_tasks_fusion_datamodules_kwcoco_dataset_full_KWCocoVideoDataset___getitem__1():
    """
    converted from /home/joncrall/code/watch/geowatch/tasks/fusion/datamodules/_orig_kwcoco_dataset_full.py::KWCocoVideoDataset.__getitem__:1
    """
    coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
    sampler = ndsampler.CocoSampler(coco_dset)
    channels = 'B10|B8a|B1|B8'
    self = KWCocoVideoDataset(sampler, time_dims=5, window_dims=(530, 610),
                              channels=channels, dist_weights=1,
                              temporal_dropout=0.5)
    item = self[0]
    canvas = self.draw_item(item)
    canvas


def test_watch_tasks_fusion_datamodules_kwcoco_dataset_full_KWCocoVideoDataset___getitem__2():
    """
    converted from /home/joncrall/code/watch/geowatch/tasks/fusion/datamodules/_orig_kwcoco_dataset_full.py::KWCocoVideoDataset.__getitem__:2
    """
    coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
    sampler = ndsampler.CocoSampler(coco_dset)
    channels = 'B10|B8|B1'
    self = KWCocoVideoDataset(sampler, time_dims=4, window_dims=(96, 96),
                              channels=channels, neg_to_pos_ratio=0.1)
    item = self[-1]
    canvas = self.draw_item(item)  # NOQA


def test_oob_target():
    import ubelt as ub
    coco_dset = geowatch.coerce_kwcoco('vidshapes8')
    sampler = ndsampler.CocoSampler(coco_dset)
    self = KWCocoVideoDataset(sampler, time_dims=1, window_dims=(128, 128),
                              channels=None)
    target = {
        'main_idx': 0,
        'gids': [2],
        'space_slice':  (
            # slice(-155.0, 357.0, None),
            # slice(-256.0, 256.0, None)
            # slice(20.0, 350, None),
            # slice(20.0, 130, None)
            slice(-200.0, 350, None),
            slice(-200.0, 350, None)
        ),
        'allow_augment': False,
    }
    index = target
    item = self[index]
    summary = self.summarize_item(item)
    print(f'summary = {ub.urepr(summary, nl=-1)}')

    s1 = h1, w1, c1 = item['frames'][0]['class_ohe'].shape
    s2 = c2, h2, w2 = item['frames'][0]['modes']['r|g|b'].shape
    s3 = h3, w3 = item['frames'][0]['saliency'].shape
    print(s1)
    print(s2)
    print(s3)
    assert h1 == h2
    assert w1 == w2

    if 0:
        import numpy as np
        import kwplot
        kwplot.autompl()
        canvas = self.draw_item(item)  # NOQA
        kwplot.imshow(canvas, pnum=(1, 3, 1), fnum=1)

        canvas = self.sampler.dset.coco_image(target['gids'][0]).imdelay().finalize()
        kwplot.imshow(canvas, pnum=(1, 3, 2), fnum=1)

        canvas = item['frames'][0]['modes']['r|g|b'].numpy().transpose(1, 2, 0).astype(np.uint8)
        kwplot.imshow(canvas, pnum=(1, 3, 3), fnum=1)


def test_resolution_on_nongeo_dataset():
    """
    The resolution parameter should help determine a scale factor on
    non-geo datasets
    """
    import ubelt as ub
    coco_dset = geowatch.coerce_kwcoco('vidshapes8')
    sampler = ndsampler.CocoSampler(coco_dset)
    self = KWCocoVideoDataset(
        sampler,
        time_dims=1,
        window_dims=(128, 128),
        channels=None,
        window_resolution=1,
        input_resolution=0.5,
        output_resolution=0.5,
        use_grid_positives=1,
        use_centered_positives=0,
    )
    index = 0
    item = self[index]
    target = item['target']
    summary = self.summarize_item(item)
    print(f'summary = {ub.urepr(summary, nl=-1)}')

    s1 = h1, w1, c1 = item['frames'][0]['class_ohe'].shape
    s2 = c2, h2, w2 = item['frames'][0]['modes']['r|g|b'].shape
    s3 = h3, w3 = item['frames'][0]['saliency'].shape
    print(s1)
    print(s2)
    print(s3)
    assert h1 == h2
    assert w1 == w2

    if 1:
        import numpy as np
        import kwplot
        kwplot.autompl()
        canvas = self.draw_item(item)  # NOQA
        kwplot.imshow(canvas, pnum=(1, 3, 1), fnum=1)

        canvas = self.sampler.dset.coco_image(target['gids'][0]).imdelay().finalize()
        kwplot.imshow(canvas, pnum=(1, 3, 2), fnum=1)

        canvas = item['frames'][0]['modes']['r|g|b'].numpy().transpose(1, 2, 0).astype(np.uint8)
        kwplot.imshow(canvas, pnum=(1, 3, 3), fnum=1)


def test_nonlocal_perframe_classification_labels():
    coco_dset = geowatch.coerce_kwcoco('vidshapes8')
    sampler = ndsampler.CocoSampler(coco_dset)
    self = KWCocoVideoDataset(
        sampler,
        time_dims=1,
        window_dims=(128, 128),
        channels=None,
        window_resolution=1,
        input_resolution=0.5,
        output_resolution=0.5,
        use_grid_positives=1,
        use_centered_positives=0,
    )
    self.requested_tasks['nonlocal_class'] = True
    self.disable_augmenter = True
    index = 0
    item = self[index]
    # ensure the nonlocal class one-hot-embedding exists
    nonlocal_class_ohe = item['frames'][0]['nonlocal_class_ohe']
    import torch
    assert torch.is_tensor(nonlocal_class_ohe)
