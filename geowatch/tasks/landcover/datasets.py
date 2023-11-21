import logging
import itertools
import kwcoco
import kwimage
import numpy as np
import torch.utils.data
from geowatch.utils import util_kwimage

log = logging.getLogger(__name__)


class _CocoTorchDataset(torch.utils.data.Dataset):
    """
    Base dataset for landcover task
    """

    def __init__(self, dset):
        self.dset = kwcoco.CocoDataset.coerce(dset)
        self.gids = sorted(list(filter(self._include, self.dset.imgs.keys())))

    def __len__(self):
        return len(self.gids)

    def __getitem__(self, idx):
        gid = self.gids[idx]
        img_info = self.dset.imgs[gid].copy()
        try:
            img_info['imgdata'] = self._load(gid)
        except Exception as ex:
            raise Exception('Unable to load image {}'.format(gid)) from ex
        return img_info

    def _include(self, gid):
        """
        Args:
            gid:
        Returns: True to include the given image in this dataset.  False to exclude.
        """
        return True

    def _load(self, img_info):
        """
        Load an image and return a numpy array.
        """
        raise NotImplementedError('subclass must override _load')

    def _load_channels_stacked(self, gid, channels_list, resolution):
        channel_images = [self._try_load_channel(gid, channels, resolution) for channels in channels_list]

        # size of largest channel from list
        dsize = max(img.shape for img in channel_images)
        dsize = (dsize[1], dsize[0])

        channel_images = [
            imresize(img, dsize=dsize, interpolation='bilinear')
            for img in channel_images
        ]
        img = np.dstack(channel_images).astype(np.float32)
        return img

    def _try_load_channel(self, gid, channels, resolution):
        if isinstance(channels, (list, tuple)):
            ex = None
            for chan in channels:
                try:
                    return self._try_load_channel(gid, chan, resolution)
                except Exception as e:
                    ex = e
            raise Exception('Unable to load any channels {} from image {}: {}'.format(channels, gid, str(ex))) from ex
        else:
            try:
                coco_img = self.dset.coco_image(gid)

                imdata = coco_img.imdelay(channels, space='image', resolution=resolution).finalize(nodata='float')
                return imdata
            except Exception as ex:
                img = self.dset.imgs[gid]
                actual_channels = img.get('channels', [aux.get('channels') for aux in img.get('auxiliary', [])])
                raise Exception(
                    'Unable to load {} from {} image with channels {}'.format(
                        channels,
                        img['sensor_coarse'],
                        actual_channels
                    )) from ex


class S2Dataset(_CocoTorchDataset):
    """
    Load S2 images and stack.
    """

    def __init__(self, dset):
        self.channels_list = [
            'coastal',
            'blue',
            'green',
            'red',
            'B05',
            'B06',
            'B07',
            'nir',
            'B8A',
            'B09',
            'cirrus',
            'swir16',
            'swir22'
        ]
        super(S2Dataset, self).__init__(dset)

    def _include(self, gid):
        sensor_type = self.dset.imgs[gid]['sensor_coarse']
        available_channels = [x["channels"] for x in self.dset.imgs[gid]["auxiliary"]]
        # Needed to handled time-averaged input images (which combine
        # all channels into a single image)
        available_channels = set(itertools.chain(*[c.split('|') for c in available_channels]))
        has_valid_sensor = sensor_type == 'S2'
        has_valid_channels = set(self.channels_list).issubset(available_channels)
        return has_valid_sensor and has_valid_channels

    def _load(self, gid):
        img = self._load_channels_stacked(gid, self.channels_list, resolution=10)
        is_samecolor = util_kwimage.find_samecolor_regions(img[:, :, 0], scale=0.4,
                                                           min_region_size=49, values={0})
        img[is_samecolor > 0] = np.nan
        img[img == -9999] = np.nan
        return img


class WVDataset(_CocoTorchDataset):
    """
    Load WorldView images and stack.
    """

    def __init__(self, dset):
        self.channels_list = [
            'coastal',
            'blue',
            'green',
            'yellow',
            'red',
            'rededge',
            'nir08',
            'nir09'
        ]
        super(WVDataset, self).__init__(dset)

    def _include(self, gid):
        sensor_type = self.dset.imgs[gid]['sensor_coarse']
        available_channels = [x["channels"] for x in self.dset.imgs[gid]["auxiliary"]]
        # Needed to handled time-averaged input images (which combine
        # all channels into a single image)
        available_channels = set(itertools.chain(*[c.split('|') for c in available_channels]))
        has_valid_sensor = sensor_type == 'WV'
        has_valid_channels = set(self.channels_list).issubset(available_channels)
        return has_valid_sensor and has_valid_channels

    def _load(self, gid):
        img = self._load_channels_stacked(gid, self.channels_list, resolution=2)
        is_samecolor = util_kwimage.find_samecolor_regions(img[:, :, 0], scale=0.4,
                                                           min_region_size=49, values={0})
        img[is_samecolor > 0] = np.nan
        img[img == -9999] = np.nan
        return img


def imresize(img, **kwargs):
    if kwargs.get('dsize') == (img.shape[1], img.shape[0]):
        return img
    return kwimage.imresize(img, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dset", type=str, required=True)
    args = parser.parse_args()

    coco_dset = WVDataset(args.coco_dset)
    print(len(coco_dset))

    img = coco_dset._load(coco_dset.gids[0])
    print(img.shape)
