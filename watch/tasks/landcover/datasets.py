import logging
from copy import deepcopy

import kwcoco
import kwimage
import numpy as np
import torch.utils.data

log = logging.getLogger(__name__)


class _CocoTorchDataset(torch.utils.data.Dataset):

    def __init__(self, dset):
        if isinstance(dset, kwcoco.CocoDataset):
            self.dset = dset
        else:
            self.dset = kwcoco.CocoDataset(dset)

        self.gids = sorted(list(filter(self._include, self.dset.imgs.keys())))

    def __len__(self):
        return len(self.gids)

    def __getitem__(self, idx):
        gid = self.gids[idx]
        img_info = deepcopy(self.dset.imgs[gid])
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

    def _load_channels_stacked(self, gid, channels_list):
        channel_images = [self._try_load_channel(gid, channels) for channels in channels_list]

        # size of largest channel from list
        dsize = max(img.shape for img in channel_images)
        dsize = (dsize[1], dsize[0])

        channel_images = [
            imresize(img, dsize=dsize, interpolation='linear')
            for img in channel_images
        ]
        img = np.dstack(channel_images)
        return img

    def _try_load_channel(self, gid, channels):
        if isinstance(channels, (list, tuple)):
            ex = None
            for chan in channels:
                try:
                    return self._try_load_channel(gid, chan)
                except Exception as e:
                    ex = e
            raise Exception('Unable to load any channels {} from image {}: {}'.format(channels, gid, str(ex))) from ex
        else:
            try:
                return self.dset.load_image(gid, channels)
            except Exception as ex:
                img = self.dset.imgs[gid]
                actual_channels = img.get('channels', [aux.get('channels') for aux in img.get('auxiliary', [])])
                raise Exception(
                    'Unable to load {} from {} image with channels {}'.format(
                        channels,
                        img['sensor_coarse'],
                        actual_channels
                    )) from ex


class L8asWV3Dataset(_CocoTorchDataset):
    """
    Load L8 images and stack them to look like WV3 images.
    """

    def _include(self, gid):
        return self.dset.imgs[gid]['sensor_coarse'] == 'L8'

    def _load(self, gid):
        channels_list = [
            'coastal',
            'blue',
            'green',
            'green',  # or pan
            'red',
            'red',
            'nir',
            'nir'
        ]
        return self._load_channels_stacked(gid, channels_list)


class S2asWV3Dataset(_CocoTorchDataset):
    """
    Load S2 images and stack them to look like WV3 images.
    """

    def _include(self, gid):
        return self.dset.imgs[gid]['sensor_coarse'] == 'S2'

    def _load(self, gid):
        channels_list = [
            'coastal',
            'blue',
            'green',
            'green',  # wrong, S2 doesn't cover this wavelength
            'red',
            'B06',
            ('B08', 'B8A', 'B07'),
            'B09'
        ]

        return self._load_channels_stacked(gid, channels_list)


class S2Dataset(_CocoTorchDataset):
    """
    Load S2 images an stack.
    """

    def _include(self, gid):
        return self.dset.imgs[gid]['sensor_coarse'] == 'S2'

    def _load(self, gid):
        all_channels = [
            'coastal',
            'blue',
            'green',
            'red',
            'B05',
            'B06', 'B07', 'nir',
            ('B8A', 'B09'),  # some images don't have B8A.  Use B09 as closest substitute
            'B09',
            'cirrus',
            'swir16',
            ('swir22', 'swir16')
        ]

        return self._load_channels_stacked(gid, all_channels)


class S2L8CommonChannelsDataset(_CocoTorchDataset):
    """
    Load the channels tha S2 and L8 have in common as a stack.
    """

    def _include(self, gid):
        return self.dset.imgs[gid]['sensor_coarse'] in ('S2', 'L8')

    def _load(self, gid):
        all_channels = [
            'blue', 'green', 'red',
            ('B8A', 'B09', 'nir'),
            'swir16',
            ('swir22', 'swir16')
        ]

        return self._load_channels_stacked(gid, all_channels)


def imresize(img, **kwargs):
    if kwargs.get('dsize') == (img.shape[1], img.shape[0]):
        return img

    return kwimage.imresize(img, **kwargs)
