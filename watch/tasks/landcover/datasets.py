import logging
# from copy import deepcopy
import kwcoco
import kwimage
import numpy as np
import torch.utils.data

log = logging.getLogger(__name__)


class _CocoTorchDataset(torch.utils.data.Dataset):
    """
    Base dataset for landcover and depth tasks
    """

    def __init__(self, dset):
        self.dset = kwcoco.CocoDataset.coerce(dset)

        self.gids = sorted(list(filter(self._include, self.dset.imgs.keys())))

    def __len__(self):
        return len(self.gids)

    def __getitem__(self, idx):
        gid = self.gids[idx]
        # should not need a deep copy here, shallow should be fine.
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

    def _load_channels_stacked(self, gid, channels_list):
        channel_images = [self._try_load_channel(gid, channels) for channels in channels_list]

        # size of largest channel from list
        dsize = max(img.shape for img in channel_images)
        dsize = (dsize[1], dsize[0])

        channel_images = [
            imresize(img, dsize=dsize, interpolation='linear')
            for img in channel_images
        ]
        img = np.dstack(channel_images).astype(np.float32)

        # set no data to nan
        img[img == -9999] = np.nan
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
                # return self.dset.load_image(gid, channels)
                # imdata = self.dset.load_image(gid, channels)
                coco_img = self.dset.coco_image(gid)
                imdata = coco_img.delay(channels, space='image').finalize(nodata='float')
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
    Load S2 images and stack.
    """

    def _include(self, gid):
        return self.dset.imgs[gid]['sensor_coarse'] == 'S2'

    def _load(self, gid):
        channels_list = [
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

        return self._load_channels_stacked(gid, channels_list)


def imresize(img, **kwargs):
    if kwargs.get('dsize') == (img.shape[1], img.shape[0]):
        return img

    return kwimage.imresize(img, **kwargs)
