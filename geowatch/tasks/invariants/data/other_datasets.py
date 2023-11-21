"""
Previously part of "datasets", but moved here because they are not used
in the main code path.
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import albumentations as A
from torchvision import transforms
import kwcoco
import kwimage
import random
from pandas import read_csv
from geowatch.tasks.invariants.utils.read_sentinel_images import read_sentinel_img_trio


class kwcoco_dataset(Dataset):
    S2_l2a_channel_names = [
        'B02.tif', 'B01.tif', 'B03.tif', 'B04.tif', 'B05.tif', 'B06.tif', 'B07.tif', 'B08.tif', 'B09.tif', 'B11.tif', 'B12.tif', 'B8A.tif'
    ]
    S2_channel_names = [
        'coastal', 'blue', 'green', 'red', 'B05', 'B06', 'B07', 'nir', 'B09', 'cirrus', 'swir16', 'swir22', 'B8A'
    ]
    L8_channel_names = [
        'coastal', 'lwir11', 'lwir12', 'blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'pan', 'cirrus'
    ]

    def __init__(self, coco_dset, sensor=['S2', 'L8'], bands=['shared'], patch_size=64, change_labels=False, display=False, mode='train'):
        # initialize dataset
        self.dset = kwcoco.CocoDataset.coerce(coco_dset)
        self.images = self.dset.images()
        self.change_labels = change_labels
        self.annotations = self.dset.annots
        self.display = display

        # handle if there are multiple sensors
        if type(sensor) is not list:
            sensor = [sensor]
        if type(bands) is not list:
            bands = [bands]
        print('Using sensors:', sensor)
        print('Using bands:', bands)

        if 'sensor_coarse' in self.images._id_to_obj[self.images._ids[0]].keys():
            # get available sensors
            # avail_sensors = set(self.images.lookup('sensor_coarse'))
            # filter images by desired sensor
            self.images = self.images.compress([x in sensor for x in self.images.lookup('sensor_coarse')])
            assert self.images
        # else:
        #     avail_sensors = None

        # get image ids and videos
        self.dset_ids = self.images.gids
        self.videos = [x['id'] for x in self.dset.videos().objs]

        # get all available channels
        all_channels = [ aux.get('channels', None) for aux in self.dset.index.imgs[self.images._ids[0]].get('auxiliary', []) ]
        if 'r|g|b' in all_channels:
            all_channels.remove('r|g|b')
        self.channels = []
        # no channels selected
        if len(bands) < 1:
            raise ValueError(f'bands must be specified. Options are {", ".join(all_channels)}, shared, or all')
        # all channels selected
        elif len(bands) == 1:
            if bands[0].lower() == 'all':
                self.channels = all_channels
            elif bands[0].lower() == 'shared':
                self.channels = ['red', 'green', 'blue', 'nir', 'swir16', 'swir22']
            elif bands[0] == 'r|g|b':
                self.channels.append('r|g|b')
            # subset of channels selected
        else:
            for band in bands:
                if band in all_channels:
                    self.channels.append(band)
                else:
                    raise ValueError(f'\'{band}\' not recognized as an available band. Options are {", ".join(all_channels)}, or all')

        # define augmentations
        self.transforms = A.Compose([
                A.RandomCrop(height=patch_size, width=patch_size),
                A.RandomRotate90(p=1)
                ],
                additional_targets={'image2': 'image', 'seg1': 'mask', 'seg2': 'mask'})

        self.transforms2 = A.Compose([
                A.RandomCrop(height=patch_size, width=patch_size),
                A.RandomRotate90(p=1),
                A.HorizontalFlip(p=1)],
            additional_targets={'image2': 'image'})

        self.transforms3 = A.Compose([A.OneOf([
                        A.MotionBlur(p=1),
                        A.Blur(blur_limit=3, p=1),
                    ], p=0.9),
                    A.GaussNoise(var_limit=.002),
                    A.RandomBrightnessContrast(brightness_limit=.3, contrast_limit=.3, brightness_by_max=False, always_apply=True)
                ])
        self.num_channels = len(self.channels)
        self.mode = mode

    def __len__(self,):
        return len(self.dset_ids)

    def get_img(self, idx, device=None):
        image_id = self.dset_ids[idx]
        image_info = self.dset.index.imgs[image_id]
        image = self.dset.delayed_load(image_id, channels=self.channels, space='video').finalize(no_data='float').astype(np.float32)
        image = torch.tensor(image)
        if device:
            image = image.to(device)
        # normalize
        if image.std() != 0.0:
            image = (image - image.mean()) / image.std()
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image_id, image_info, image

    def __getitem__(self, idx):
        # get image1 id and the video it is associated with
        img1_id = self.dset_ids[idx]
        if self.mode == 'test':
            img1_info = self.dset.index.imgs[img1_id]
        else:
            img1_info = torch.tensor([])

        img_obj1 : dict = self.dset.index.imgs[img1_id]
        video : int = img_obj1['video_id']

        # randomly select image2 id from the same video (could be before or after image1)
        # make sure image2 is not image1 and image2 is in the set of filtered images by desired sensor
        img2_id = img1_id
        while img2_id == img1_id or img2_id not in self.dset_ids:
            img2_id = random.choice(self.dset.index.vidid_to_gids[video])

        img_obj2 : dict = self.dset.index.imgs[img2_id]

        # get frame indices for each image (used to determine which image was captured first)
        frame_index1 = img_obj1['frame_index']
        frame_index2 = img_obj2['frame_index']
        # get sensors
        im1_sensor = img_obj1['sensor_coarse']
        im2_sensor = img_obj2['sensor_coarse']

        # load images
        img1 = self.dset.delayed_load(img1_id, channels=self.channels, space='video').finalize(no_data='float').astype(np.float32)
        img2 = self.dset.delayed_load(img2_id, channels=self.channels, space='video').finalize(no_data='float').astype(np.float32)

        if not self.change_labels:
            # transformations
            max1 = img1.max()
            max2 = img2.max()
            transformed = self.transforms(image=img1.copy() / max1, image2=img2.copy() / max2)
            transformed2 = self.transforms2(image=img1.copy() / max1)
            img1 = transformed['image'] * max1
            img2 = transformed['image2'] * max2

            if self.display:
                if self.num_channels == 3:
                    display_image1 = img1
                    display_image2 = img2
                else:

                    display_image1 = img1[:, :, [3, 2, 1]]
                    display_image2 = img2[:, :, [3, 2, 1]]
                display_image1 = (2 + display_image1) / 6
                display_image2 = (2 + display_image2) / 6
                display_image1 = torch.tensor(display_image1).permute(2, 0, 1)
                display_image2 = torch.tensor(display_image2).permute(2, 0, 1)
            else:
                display_image1 = torch.tensor([])
                display_image2 = torch.tensor([])

            img3 = transformed2['image'] * max1
            img4 = self.transforms3(image=img1.copy() / img1.max())['image'] * img1.max()
            # convert to tensors

            img1std = np.nanstd(img1)
            if img1std != 0.:
                img1 = (img1 - np.nanmean(img1)) / img1std
            else:
                img1 = np.zeros_like(img1)
            img2std = np.nanstd(img2)
            if img2std != 0.:
                img2 = (img2 - np.nanmean(img2)) / img2std
            else:
                img2 = np.zeros_like(img2)
            img3std = np.nanstd(img3)
            if img3std != 0.:
                img3 = (img3 - np.nanmean(img3)) / img3std
            else:
                img3 = np.zeros_like(img3)
            img4std = np.nanstd(img4)
            if img4std != 0.:
                img4 = (img4 - np.nanmean(img4)) / img4std
            else:
                img4 = np.zeros_like(img4)

            img1 = np.nan_to_num(img1)
            img2 = np.nan_to_num(img2)
            img3 = np.nan_to_num(img3)
            img4 = np.nan_to_num(img4)

            img1 = torch.tensor(img1).permute(2, 0, 1)
            img2 = torch.tensor(img2).permute(2, 0, 1)
            img3 = torch.tensor(img3).permute(2, 0, 1)
            img4 = torch.tensor(img4).permute(2, 0, 1)

            return {
                'image1': img1.float(),
                'image2': img2.float(),
                'offset_image1': img3.float(),
                'augmented_image1': img4.float(),
                'time_sort_label': float(frame_index1 < frame_index2),
                'date1': (frame_index1, frame_index1),
                'date2': (frame_index2, frame_index2),
                'display_image1': display_image1,
                'display_image2': display_image2,
                'sensor_image1': im1_sensor,
                'sensor_image2': im2_sensor,
                'img1_id': img1_id,
                'img1_info': img1_info
            }

        else:
            if frame_index1 > frame_index2:
                img1, img2 = img2, img1
                img1_id, img2_id = img2_id, img1_id
                img_obj1, img_obj2 = img_obj2, img_obj1

            aids1 = self.dset.index.gid_to_aids[img1_id]
            aids2 = self.dset.index.gid_to_aids[img2_id]
            dets1 = kwimage.Detections.from_coco_annots(
                self.dset.annots(aids1).objs, dset=self.dset)
            dets2 = kwimage.Detections.from_coco_annots(
                self.dset.annots(aids2).objs, dset=self.dset)

            vid_from_img1 = kwimage.Affine.coerce(img_obj1['warp_img_to_vid'])
            vid_from_img2 = kwimage.Affine.coerce(img_obj2['warp_img_to_vid'])

            dets1 = dets1.warp(vid_from_img1)
            dets2 = dets2.warp(vid_from_img2)

            # bbox = dets.data['boxes'].data
            segmentation1 = dets1.data['segmentations'].data
            segmentation2 = dets2.data['segmentations'].data
            category_id1 = [dets1.classes.idx_to_id[cidx] for cidx in dets1.data['class_idxs']]
            category_id2 = [dets2.classes.idx_to_id[cidx] for cidx in dets2.data['class_idxs']]

            img_dims = (img1.shape[0], img1.shape[1])

            combined1 = []

            for sseg, cid in zip(segmentation1, category_id1):
                assert cid > 0
                np_mask = sseg.to_mask(dims=img_dims).data.astype(float) * cid
                mask1 = torch.from_numpy(np_mask)
                combined1.append(mask1.unsqueeze(0))

            if combined1:
                overall_mask1 = torch.max(torch.cat(combined1, dim=0), dim=0)[0]
            else:
                overall_mask1 = np.zeros_like(img1[:, :, 0])

            combined2 = []

            for sseg, cid in zip(segmentation2, category_id2):
                assert cid > 0
                np_mask = sseg.to_mask(dims=img_dims).data.astype(float) * cid
                mask2 = torch.from_numpy(np_mask)
                combined2.append(mask2.unsqueeze(0))

            if combined2:
                overall_mask2 = torch.max(torch.cat(combined2, dim=0), dim=0)[0]
            else:
                overall_mask2 = np.zeros_like(img2[:, :, 0])

            transformed = self.transforms(image=img1, image2=img2, seg1=np.array(overall_mask1), seg2=np.array(overall_mask2))
            img1 = transformed['image']
            img2 = transformed['image2']
            segmentation1 = transformed['seg1']
            segmentation2 = transformed['seg2']

            img1 = torch.tensor(img1).permute(2, 0, 1)
            img2 = torch.tensor(img2).permute(2, 0, 1)

            if self.display:
                if self.num_channels == 3:
                    display_image1 = img1
                    display_image2 = img2
                else:
                    display_image1 = img1[[3, 2, 1], :, :]
                    display_image2 = img2[[3, 2, 1], :, :]
                display_image1 = (2 + display_image1) / 6
                display_image2 = (2 + display_image2) / 6
            else:
                display_image1 = torch.tensor([])
                display_image2 = torch.tensor([])

            segmentation1 = torch.tensor(segmentation1)
            segmentation2 = torch.tensor(segmentation2)
            change_map = torch.clamp(segmentation2 - segmentation1, 0, 1)

            img1 = np.nan_to_num(img1)
            img2 = np.nan_to_num(img2)

            return {
                'image1': img1.float(),
                'image2': img2.float(),
                'segmentation1': segmentation1,
                'segmentation2': segmentation2,
                # 'categories1': category_id1,
                # 'categories2': category_id2,
                'segmentation1': segmentation1,
                'segmentation2': segmentation2,
                'change_map': change_map,
                'display_image1': display_image1,
                'display_image2': display_image2,
                'sensor_image1': im1_sensor,
                'sensor_image2': im2_sensor,
                'img1_id': img1_id,
                'img1_info': img1_info
            }


class Onera(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self,
                 data_folder='/localdisk0/SCRATCH/watch/onera/',
                 train=True,
                 patch_size=96,
                 num_channels=13,
                 multihead=False,
                 display=False,
                 class_weight=1,
                 randomize_order=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.display = display
        self.randomize_order = randomize_order
        # basics
        self.path = data_folder
        self.num_channels = num_channels
        self.img_subpath = 'images/'
        self.train = train

        if self.train:
            fname = 'train.txt'
            self.label_subpath = 'train_labels/'
            self.transforms1 = A.Compose([
                                                A.HorizontalFlip(p=0.5),
                                                A.RandomRotate90(p=.99),
                                                A.RandomCrop(patch_size, patch_size)
                                            ],
                                            additional_targets={'image1': 'image', 'image2': 'image', 'mask1': 'mask'})
        else:
            fname = 'test.txt'
            self.label_subpath = 'test_labels/'
            self.transforms1 = A.Compose([A.NoOp()],
                                            additional_targets={'image1': 'image', 'image2': 'image', 'mask1': 'mask'})

        self.multihead = multihead
        if multihead:
            if train:
                self.transforms2 = A.Compose([
                                                A.RandomCrop(height=patch_size, width=patch_size),
                                                A.RandomRotate90(p=0.5),
                                                A.HorizontalFlip(p=.75),
                                                A.VerticalFlip(p=.75)
                                                ],
                                                additional_targets={'image2': 'image'})
            else:
                self.transforms2 = A.Compose([
                                                # A.RandomCrop(height=patch_size, width=patch_size),
                                                A.HorizontalFlip(p=.75),
                                                A.VerticalFlip(p=.75)
                                                ],
                                                additional_targets={'image2': 'image'})
            self.transforms3 = A.Compose([
                    A.Blur(p=.3),
                    A.RandomBrightnessContrast(always_apply=True)
            ])

        self.to_tensor = transforms.ToTensor()

        self.loc_names = read_csv(self.path + self.img_subpath + fname).columns

        self.num_channels = num_channels

    def __len__(self):
        if self.train:
            return 2560
        else:
            return 10

    def __getitem__(self, idx):
        if self.train:
            idx = idx % 14

        loc_name = self.loc_names[idx]
        img1, img2, cm = read_sentinel_img_trio(self.path + self.img_subpath + loc_name, self.path + self.label_subpath + loc_name, self.num_channels, True)

        img1 = (img1 - img1.mean()) / img1.std()
        img2 = (img2 - img2.mean()) / img2.std()

        cm = 1 * np.array(cm)

        transformed = self.transforms1(image=img1, image2=img2, mask=cm)
        img1 = transformed['image']
        img2 = transformed['image2']
        change_map = transformed['mask']

        if self.multihead:
            transformed2 = self.transforms2(image=img1)
            img3 = transformed2['image']
            img3 = self.to_tensor(img3)

            transformed3 = self.transforms3(image=img1)
            img4 = transformed3['image']
            img4 = self.to_tensor(img4)

        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)

        date1 = (0, 0)
        date2 = (1, 1)

        if self.randomize_order:
            label = random.choice([0, 1])
        else:
            label = 1

        if not label:
            img1, img2 = img2, img1
            date1, date2 = date2, date1

        if self.display:
            if self.num_channels == 3:
                display_image1 = img1
                display_image2 = img2
            elif self.num_channels == 13:
                display_image1 = img1[[3, 2, 1], : , :]
                display_image2 = img2[[3, 2, 1], : , :]
            else:
                display_image1 = img1[[2, 1, 0], :, :]
                display_image2 = img2[[2, 1, 0], :, :]
            display_image1 = (2 + display_image1) / 3
            display_image2 = (2 + display_image2) / 3
        else:
            display_image1 = torch.tensor([])
            display_image2 = torch.tensor([])

        if not self.multihead:
            return {'image1': img1.float().contiguous(),
                    'image2': img2.float().contiguous(),
                    'change_map': change_map,
                    'label': label,
                    'date1': date1,
                    'date2': date2,
                    'display_image1': display_image1,
                    'display_image2': display_image2,
                    'time_steps': torch.tensor([0, 1])}

        else:
            return {'image1': img1.float(),
                    'image2': img2.float(),
                    'offset_image': img3.float(),
                    'augmented_image': img4.float(),
                    'change_map': change_map,
                    'label': label,
                    'date1': date1,
                    'date2': date2,
                    'display_image1': display_image1.contiguous(),
                    'display_image2': display_image2.contiguous(),
                    'time_steps': torch.tensor([0, 1])}


class SpaceNet7(Dataset):
    normalize_params = [[0.16198677, 0.22665408, 0.1745371], [0.06108317, 0.06515977, 0.04128775]]

    def __init__(self,
                    patch_size=[128, 128],
                    splits='satellite_sort/data/spacenet/splits_unmasked/',  # ### unmasked images
                    train=True,
                    normalize=True,
                    yearly=True,
                    display=False):

        self.display = display
        self.train = train
        self.yearly = yearly
        self.crop = A.Compose([A.RandomCrop(height=patch_size[0], width=patch_size[1])], additional_targets={'image2': 'image', 'mask1': 'mask', 'mask2': 'mask'})
        if self.train:
            self.rotate = A.Compose([A.RandomRotate90()], additional_targets={'image2': 'image',
                                                                               'mask1': 'mask',
                                                                               'mask2': 'mask'})
            self.transforms = A.Compose([
                A.GaussianBlur(),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                        ],
                                additional_targets={'image2': 'image'}
            )
        else:
            self.transforms = None

        if train:
            self.images = []
            with open(os.path.join(splits, 'spacenet7_train_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))

        else:
            self.images = []
            with open(os.path.join(splits, 'spacenet7_val_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))

        self.length = len(self.images)
        self.normalize = normalize
        self.normalization = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)

        self.time_adjust = [-1, 1]

    def __len__(self):
        return (self.length)

    def __getitem__(self, idx):

        im1_path = self.images[idx]
        date1 = (int(im1_path[-47:-43]), int(im1_path[-42:-40]))

        if self.yearly:
            ###Choose image2 as close to one year spread apart as possible
            time_adjust = random.choice(self.time_adjust)
            im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) + time_adjust) + im1_path[-43:]

            if not os.path.exists(im2_path):
                im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) - time_adjust) + im1_path[-43:]

                if not os.path.exists(im2_path):
                    im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) - time_adjust) + im1_path[-43]

                    if not os.path.exists(im2_path):
                        im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) - time_adjust) + im1_path[-43:]

                        x = 0
                        while not os.path.exists(im2_path):
                            x += 1
                            if not os.path.exists(im2_path):
                                im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) - time_adjust) + im1_path[-43:-41] + str(int(im1_path[-41]) + x * time_adjust) + im1_path[-40:]
                                if not os.path.exists(im2_path):
                                    im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) - time_adjust) + im1_path[-43:-41] + str(int(im1_path[-41]) - x * time_adjust) + im1_path[-40:]

                                    if not os.path.exists(im2_path):
                                        im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) + time_adjust) + im1_path[-43:-41] + str(int(im1_path[-41]) + x * time_adjust) + im1_path[-40:]
                                        if not os.path.exists(im2_path):
                                            im2_path = im1_path[:-47] + str(int(im1_path[-47:-43]) + time_adjust) + im1_path[-43:-41] + str(int(im1_path[-41]) - x * time_adjust) + im1_path[-40:]

        else:
            im_directory, _ = os.path.split(self.images[idx])
            date1 = (int(self.images[idx][-47:-43]), int(self.images[idx][-42:-40]))
            date2 = date1
            while date2 == date1:
                idx2 = torch.randint(len(os.listdir(im_directory)), (1,))[0]
                im2_path = os.path.join(im_directory, sorted(os.listdir(im_directory))[idx2])

        date2 = (int(im2_path[-47:-43]), int(im2_path[-42:-40]))

        '''TO DO: Convert tiffile package commands to kwimage'''
        image = (kwimage.imread(im1_path)).astype("float32")
        image2 = (kwimage.imread(im2_path)).astype("float32")

        cloud_mask1 = image[:, :, 3]
        cloud_mask2 = image2[:, :, 3]

        image = image[:, :, :3]
        image2 = image2[:, :, :3]

        crop = self.crop(image=image, image2=image2, mask1=cloud_mask1, mask2=cloud_mask2)

        image = crop['image']
        image2 = crop['image2']
        cloud_mask1 = crop['mask1']
        cloud_mask2 = crop['mask2']

        if self.display:
            display_image1 = image.astype('uint8')
            display_image2 = image2.astype('uint8')
        else:
            display_image1 = torch.tensor([])
            display_image2 = torch.tensor([])

        if self.normalize:
            image = self.normalization(image=image)['image']
            image2 = self.normalization(image=image2)['image']

        if self.transforms:
            transformed = self.transforms(image=image, image2=image2)
            image = transformed['image']
            image2 = transformed['image2']

            rotated = self.rotate(image=image, image2=image2)
            image = rotated['image']
            image2 = rotated['image2']

        item = {
            'image1': torch.tensor(image).permute(2, 0, 1).contiguous(),
            'image2': torch.tensor(image2).permute(2, 0, 1).contiguous(),
            'label': int(date1 < date2),
            'date1': date1,
            'date2': date2,
            'display_image1': display_image1.contiguous(),
            'display_image2': display_image2.contiguous(),
            'cloud_mask1': cloud_mask1.contiguous(),
            'cloud_mask2': cloud_mask2.contiguous(),
        }
        return item
