import numpy as np
import kwcoco
# from torchvision import transforms

# from .demo_transform import Normalize, ToTensor, ToNumpy
from .dzyne_img_util import normalizeRGB
from ..landcover.datasets import _CocoTorchDataset

WV_CHANNELS = "coastal|blue|green|yellow|red|red-edge|near-ir1|near-ir2"


class WVRgbDataset(_CocoTorchDataset):
    """
    Dataset for depth prediction

    Args:
        dset (kwcoco.CocoDataset):
            input dataset to wrap

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from geowatch.tasks.depth.datasets import *  # NOQA
        >>> import geowatch
        >>> import kwcoco
        >>> dvc_dpath = geowatch.find_dvc_dpath()
        >>> coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json'
        >>> input_dset = kwcoco.CocoDataset(coco_fpath)
        >>> self = WVRgbDataset(input_dset)
        >>> # Test that the "include" correctly filter to only WorldView
        >>> images = input_dset.images(self.gids)
        >>> assert all(s == 'WV' for s in images.lookup('sensor_coarse'))
        >>> gid = self.gids[0]
        >>> img = self._load(gid)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(kwarray.normalize(img))
        >>> kwplot.show_if_requested()
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.__imagenet_stats = {
            'mean': np.array([0.485, 0.456, 0.406])[None, None, :],
            'std': np.array([0.229, 0.224, 0.225])[None, None, :],
        }

    def _include(self, gid):
        """
        Used on init to filter to only relevant images
        """
        coco_img = self.dset.coco_image(gid)
        is_wv = coco_img.img['sensor_coarse'] == 'WV'
        has_rgb = (coco_img.channels & 'red|green|blue').numel() == 3
        return is_wv and has_rgb

    def _load(self, gid):
        coco_img = self.dset.coco_image(gid)

        # List all the channels contained in this image
        have_channels: set = coco_img.channels.fuse().as_set()

        # Work with rgb by default, but fallback on panchromatic
        # TODO: do we want to coerce panchromatic images to RGB or vis versa?
        want_channels1 = kwcoco.FusedChannelSpec.coerce('red|green|blue')
        # want_channels2 = kwcoco.FusedChannelSpec.coerce('panchromatic')
        has_rgb = want_channels1.as_set().issubset(have_channels)
        if has_rgb:
            want_channels = want_channels1
        if not has_rgb:
            raise NotImplementedError('We expected to have RGB available')
            # has_pan = want_channels2.as_set().issubset(have_channels)
            # if has_pan:
            #     want_channels = want_channels2
            # else:
            #     raise NotImplementedError(f'No Pan or RGB in {have_channels}')

        delayed = coco_img.imdelay(channels=want_channels)
        img = delayed.finalize(nodata='float')
        img = np.asarray(img).astype(np.float32)

        NODATA_HACK = True
        if NODATA_HACK:
            # Hack to handle cases where nodata is not in the geotiff metadata.
            # In these cases assume that zero is the nodata value. This is
            # not safe in general, and new outputs (March 2022 and later should
            # correct for this)
            nodata_values = {}
            for aux in coco_img.iter_asset_objs():
                aux_chan = kwcoco.FusedChannelSpec.coerce(aux['channels'])
                band_metas = aux.get('band_metas', [])
                if len(band_metas) != aux_chan.numel():
                    raise AssertionError
                for meta, chan in zip(band_metas, aux_chan.as_list()):
                    if chan in want_channels:
                        nodata_values[chan] = meta['nodata']

            if all(v is None for v in nodata_values.values()):
                # Nodata was not specified in metadata, assume it is zero
                img[img == 0] = np.nan

        img = normalizeRGB(img, (3, 97))
        img -= self.__imagenet_stats['mean']
        img /= self.__imagenet_stats['std']
        return img


class WVSuperRgbDataset(_CocoTorchDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(kwargs['dset'])
        self.__imagenet_stats = {
            'mean': np.array([0.485, 0.456, 0.406])[None, None, :],
            'std': np.array([0.229, 0.224, 0.225])[None, None, :],
        }

        self.scale = int(kwargs['scale'])
        assert (self.scale > 0)

    def _include(self, gid):
        """
        Used on init to filter to only relevant images
        """
        coco_img = self.dset.coco_image(gid)
        is_wv = coco_img.img['sensor_coarse'] == 'WV'
        has_rgb = (coco_img.channels & 'superRes_red|superRes_green|superRes_blue').numel() == 3
        return is_wv and has_rgb

    def _load(self, gid):

        # Get image to work on
        coco_img = self.dset.coco_image(gid)

        # Check for required channels
        available_channels: set = coco_img.channels.fuse().as_set()
        required_channels = kwcoco.FusedChannelSpec.coerce('superRes_red|superRes_green|superRes_blue')
        has_rgb = required_channels.as_set().issubset(available_channels)

        if has_rgb is False:
            raise NotImplementedError('We expected to have RGB available')

        # Load image using required channels
        delayed = coco_img.delay(channels=required_channels)
        img = delayed.scale(self.scale).finalize(nodata='float')  # TODO - auto-detect scaling factor using image and aux metadata
        img = np.asarray(img, dtype=np.float32)

        # Post-processing
        NODATA_HACK = True
        if NODATA_HACK:
            # Hack to handle cases where nodata is not in the geotiff metadata.
            # In these cases assume that zero is the nodata value. This is
            # not safe in general, and new outputs (March 2022 and later should
            # correct for this)
            nodata_values = {}
            for aux in coco_img.iter_asset_objs():
                aux_chan = kwcoco.FusedChannelSpec.coerce(aux['channels'])
                band_metas = aux.get('band_metas', [])
                if len(band_metas) != aux_chan.numel():
                    raise AssertionError
                for meta, chan in zip(band_metas, aux_chan.as_list()):
                    if chan in required_channels:
                        nodata_values[chan] = meta['nodata']

            if all(v is None for v in nodata_values.values()):
                # Nodata was not specified in metadata, assume it is zero
                img[img == 0] = np.nan

        img = normalizeRGB(img, (3, 97))
        img -= self.__imagenet_stats['mean']
        img /= self.__imagenet_stats['std']
        return img
