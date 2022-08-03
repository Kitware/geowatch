import torch
import kwcoco
import random
import numpy as np

import albumentations as A


class kwcoco_dataset(torch.utils.data.Dataset):
    S2_channel_names = [
        'coastal', 'blue', 'green', 'red', 'B05', 'B06', 'B07', 'nir', 'B09', 'cirrus', 'swir16', 'swir22', 'B8A'
    ]
    L8_channel_names = [
        'coastal', 'lwir11', 'lwir12', 'blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'pan', 'cirrus'
    ]
    common_channel_names = [
        'coastal', 'blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'cirrus'
    ]

    def __init__(self, coco_dset, sensor='S2', bands='all', patch_size=64, mode='train'):
        # initialize dataset
        self.dset = kwcoco.CocoDataset.coerce(coco_dset)
        self.images = self.dset.images()

        # handle if there are multiple sensors
        if sensor != 'all':
            if 'sensor_coarse' in self.images._id_to_obj[self.images._ids[0]].keys():
                # get available sensors
                avail_sensors = set(self.images.lookup('sensor_coarse'))
                # invalid sensor
                if sensor is None or sensor not in avail_sensors:
                    raise ValueError(f'sensor must be specified. Options are {", ".join(avail_sensors)}')
                # filter images by desired sensor
                else:
                    self.images = self.images.compress([x == sensor for x in self.images.lookup('sensor_coarse')])
                    assert self.images
            else:
                avail_sensors = None

        # get image ids and videos
        self.dset_ids = self.images.gids
        self.videos = [x['id'] for x in self.dset.videos().objs]

        # get all available channels
        all_channels = [ aux.get('channels', None) for aux in self.dset.index.imgs[self.images._ids[0]].get('auxiliary', []) ]
        if 'r|g|b' in all_channels:
            all_channels.remove('r|g|b')
        self.channels = []
        # no channels selected

        if bands == 'all':
            bands = '|'.join(all_channels)
        elif bands == 'S2':
            bands = '|'.join(self.S2_channel_names)
        elif bands == 'L8':
            bands = '|'.join(self.L8_channel_names)
        elif bands == 'common':
            bands = '|'.join(self.L8_channel_names)

        self.channels = kwcoco.FusedChannelSpec.coerce(bands)
        # if len(bands) < 1:
        #     raise ValueError(f'bands must be specified. Options are {", ".join(all_channels)}, or all')
        # # all channels selected
        # elif len(bands) == 1 and bands[0].lower() == 'all':
        #     self.channels = all_channels
        # # subset of channels selected
        # else:
        #     for band in bands:
        #         if band in all_channels:
        #             self.channels.append(band)
        #         else:
        #             raise ValueError(f'\'{band}\' not recognized as an available band. Options are {", ".join(all_channels)}, or all')

        # define augmentations
        self.transforms = A.Compose([
                A.RandomCrop(height=patch_size, width=patch_size),
                A.RandomRotate90(p=.999)],
            additional_targets={'image2': 'image'})

        self.transforms2 = A.Compose([
                A.RandomCrop(height=patch_size, width=patch_size),
                A.RandomRotate90(p=.999),
                A.HorizontalFlip(p=0.999)],
            additional_targets={'image2': 'image'})

        self.transforms3 = A.Compose([
                A.Blur(p=.75),
                A.RandomBrightnessContrast(brightness_by_max=False, always_apply=True)
        ])

        self.mode = mode

    def __len__(self,):
        return len(self.dset_ids)

    def get_img(self, idx, device=None):
        image_id = self.dset_ids[idx]
        image_info = self.dset.index.imgs[image_id]
        delayed = self.dset.delayed_load(
            image_id, channels=self.channels, space='video')
        image = delayed.finalize()
        real_mean = np.nanmean(image)
        real_std = np.nanstd(image)
        image = np.nan_to_num(image)
        image = image.astype(np.float32)
        image = torch.tensor(image)
        # normalize
        if real_std != 0.0:
            image = (image - real_mean) / real_std
        if device:
            image = image.to(device)
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image_id, image_info, image

    def __getitem__(self, idx):
        if self.mode == 'test':
            # hack
            return self.get_img(idx)

        # get image1 id and the video it is associated with
        img1_id = self.dset_ids[idx]
        video = [ y for y in self.videos if img1_id in self.dset.index.vidid_to_gids[y] ][0]

        # randomly select image2 id from the same video (could be before or after image1)
        # make sure image2 is not image1 and image2 is in the set of filtered images by desired sensor
        img2_id = img1_id
        while img2_id == img1_id or img2_id not in self.dset_ids:
            img2_id = random.choice(self.dset.index.vidid_to_gids[video])

        # get frame indices for each image (used to determine which image was captured first)
        frame_index1 = self.dset.index.imgs[img1_id]['frame_index']
        frame_index2 = self.dset.index.imgs[img2_id]['frame_index']

        # load images
        img1 = self.dset.delayed_load(img1_id, channels=self.channels, space='video').finalize().astype(np.float32)
        img2 = self.dset.delayed_load(img2_id, channels=self.channels, space='video').finalize().astype(np.float32)
        img1 = np.nan_to_num(img1)
        img2 = np.nan_to_num(img2)

        # normalize
        if img1.std() != 0.0:
            img1 = (img1 - img1.mean()) / img1.std()
        if img2.std() != 0.0:
            img2 = (img2 - img2.mean()) / img2.std()

        # transformations
        transformed = self.transforms(image=img1, image2=img2)
        transformed2 = self.transforms2(image=img1)

        img1 = transformed['image']
        img2 = transformed['image2']
        img3 = transformed2['image']

        transformed3 = self.transforms3(image=img1)
        img4 = transformed3['image']

        # convert to tensors
        img1 = torch.tensor(img1).permute(2, 0, 1)
        img2 = torch.tensor(img2).permute(2, 0, 1)
        img3 = torch.tensor(img3).permute(2, 0, 1)
        img4 = torch.tensor(img4).permute(2, 0, 1)

        return {
                'image1': img1.float(),
                'image2': img2.float(),
                'offset_image1': img3.float(),
                'augmented_image1': img4.float(),
                'time_sort_label': float(frame_index1 < frame_index2)}

    def num_channels(self):
        return len(self.channels)
