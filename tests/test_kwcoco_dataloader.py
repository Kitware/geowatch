"""
Originally these were doctests in the dataloader, but were removed for space.

TODO: These should be refactored into more descriptive tests that target
specific parameters in the datamodule config
"""
from watch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
import ndsampler
import kwcoco
import watch


def test_watch_tasks_fusion_datamodules_kwcoco_dataset_full_KWCocoVideoDataset():
    """
    converted from /home/joncrall/code/watch/watch/tasks/fusion/datamodules/_orig_kwcoco_dataset_full.py::KWCocoVideoDataset:1
    """
    coco_dset = watch.demo.coerce_kwcoco('watch')
    print({c.get('sensor_coarse') for c in coco_dset.images().coco_images})
    print({c.channels.spec for c in coco_dset.images().coco_images})
    sampler = ndsampler.CocoSampler(coco_dset)
    sample_shape = (2, 128, 128)
    self = KWCocoVideoDataset(sampler, sample_shape=sample_shape,
                              channels=None)
    index = 0
    item = self[index]
    canvas = self.draw_item(item)
    canvas


def test_watch_tasks_fusion_datamodules_kwcoco_dataset_full_KWCocoVideoDataset___getitem__1():
    """
    converted from /home/joncrall/code/watch/watch/tasks/fusion/datamodules/_orig_kwcoco_dataset_full.py::KWCocoVideoDataset.__getitem__:1
    """
    coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
    sampler = ndsampler.CocoSampler(coco_dset)
    channels = 'B10|B8a|B1|B8'
    sample_shape = (5, 530, 610)
    self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, dist_weights=1, temporal_dropout=0.5)
    item = self[0]
    canvas = self.draw_item(item)
    canvas


def test_watch_tasks_fusion_datamodules_kwcoco_dataset_full_KWCocoVideoDataset___getitem__2():
    """
    converted from /home/joncrall/code/watch/watch/tasks/fusion/datamodules/_orig_kwcoco_dataset_full.py::KWCocoVideoDataset.__getitem__:2
    """
    coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
    sampler = ndsampler.CocoSampler(coco_dset)
    channels = 'B10|B8|B1'
    sample_shape = (4, 96, 96)
    self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=channels, neg_to_pos_ratio=0.1)
    item = self[-1]
    canvas = self.draw_item(item)  # NOQA
