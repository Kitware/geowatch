from torch.utils.data import Dataset
import torch
import numpy as np
import os
import albumentations as A
import kwcoco
import kwimage
import random
import ubelt as ub
import tifffile as tif


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

    def __init__(self, coco_dset, sensor=['S2', 'L8'], bands=['shared'], patch_size=64, segmentation_labels=False, display=False, num_images=3, mode='train'):
        # initialize dataset
        self.dset = kwcoco.CocoDataset.coerce(coco_dset)
        self.images = self.dset.images()
        self.segmentation_labels = segmentation_labels
        self.annotations = self.dset.annots
        self.display = display

        self.num_images = num_images

        if not ub.iterable(sensor):
            sensor = [sensor]
        if not ub.iterable(bands):
            bands = [sensor]

        requested_sensors = set(sensor)

        if type(sensor) is not list:
            sensor = [sensor]
        if type(bands) is not list:
            bands = [bands]

        # handle if there are multiple sensors
        image_sensors = self.images.lookup('sensor_coarse', default=None)
        avail_sensors = set(image_sensors)
        flags = [x in requested_sensors for x in image_sensors]
        self.images = self.images.compress(flags)

        # if 'sensor_coarse' in self.images._id_to_obj[self.images._ids[0]].keys():
        #     # get available sensors
        #     # filter images by desired sensor
        #     self.images = self.images.compress([x in sensor for x in self.images.lookup('sensor_coarse')])
        #     assert(self.images)
        # else:
        #     avail_sensors = None
        print('avail_sensors:', avail_sensors)
        print('requested_sensors:', requested_sensors)
        self.dset_ids = self.images.gids
        self.videos = [x['id'] for x in self.dset.videos().objs]

        # get all available channels
        all_channels = [ aux.get('channels', None) for aux in self.dset.index.imgs[self.images._ids[0]].get('auxiliary', []) ]
        if 'r|g|b' in all_channels:
            all_channels.remove('r|g|b')
        self.channels = []
        # no channels selected
        if len(bands) < 1:
            raise ValueError(f'bands must be specified. Options are {", ".join(all_channels)}, or all')
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
        additional_targets = dict()
        for i in range(self.num_images):
            additional_targets['image{}'.format(1 + i)] = 'image'
            additional_targets['seg{}'.format(i + 1)] = 'mask'

        if mode == 'test':
            self.segmentation_labels = False
            self.transforms = A.Compose([
                A.NoOp(),
                ],
                additional_targets=additional_targets)
            self.transforms2 = A.Compose([
                    A.VerticalFlip(p=.999),
                    A.HorizontalFlip(p=.999)],
                additional_targets={'image2': 'image'})

            self.transforms3 = A.Compose([
                    A.Blur(p=.75),
                    A.RandomBrightnessContrast(brightness_by_max=False, always_apply=True)
            ])
        else:
            self.transforms = A.Compose([
                A.RandomCrop(height=patch_size, width=patch_size),
                A.RandomRotate90(p=.999)
                ],
                additional_targets=additional_targets)

        self.num_channels = len(self.channels)
        self.bands = bands

        self.mode = mode

    def __len__(self,):
        return len(self.dset_ids)

    def get_img(self, idx, device=None):
        img1_id = self.dset_ids[idx]
        video = [y for y in self.videos if img1_id in self.dset.index.vidid_to_gids[y]][0]

        # randomly select image2 id from the same video (could be before or after image1)
        # make sure image2 is not image1 and image2 is in the set of filtered images by desired sensor
        id_list = [img1_id]
        frame_index = {1: self.dset.index.imgs[img1_id]['frame_index']}
        sensor_list = [self.dset.index.imgs[img1_id]['sensor_coarse']]
        for j in range(1, self.num_images):
            new_img_id = img1_id
            while (new_img_id in id_list) or (new_img_id not in self.dset_ids):
                new_img_id = random.choice(self.dset.index.vidid_to_gids[video])
            id_list.append(new_img_id)
            frame_index[1 + j] = self.dset.index.imgs[new_img_id]['frame_index']
            sensor_list.append(self.dset.index.imgs[new_img_id]['sensor_coarse'])
        # load images
        image_dict = dict()
        for k in range(self.num_images):
            image = self.dset.delayed_load(id_list[k], channels=self.channels, space='video').finalize().astype(np.float32)

            image = np.nan_to_num(image)
            if image.std() != 0.:
                image = (image - image.mean()) / image.std()
            image_dict['img{}'.format(1 + k)] = image

        image = [torch.tensor(image_dict[key]).permute(2, 0, 1).unsqueeze(0) for key in image_dict]

        image = torch.stack(image, dim=1)
        image_info = self.dset.index.imgs[img1_id]

        image = torch.nan_to_num(image)

        if device:
            image = image.to(device)
        # normalize
        if image.std() != 0.0:
            image = (image - image.mean()) / image.std()

        return img1_id, image_info, image

    def __getitem__(self, idx):
        # get image1 id and the video it is associated with
        img1_id = self.dset_ids[idx]
        video = [y for y in self.videos if img1_id in self.dset.index.vidid_to_gids[y]][0]
        out = dict()
        # randomly select image2 id from the same video (could be before or after image1)
        # make sure image2 is not image1 and image2 is in the set of filtered images by desired sensor
        id_list = [img1_id]
        img1_info = self.dset.index.imgs[img1_id]
        frame_index = {1: img1_info['frame_index']}
        sensor_list = [img1_info['sensor_coarse']]
        date = img1_info['date_captured']
        date_list = [torch.tensor([int(date[:4]), int(date[5:7])])]
        for j in range(1, self.num_images):
            new_img_id = img1_id
            while (new_img_id in id_list) or (new_img_id not in self.dset_ids):
                new_img_id = random.choice(self.dset.index.vidid_to_gids[video])
            id_list.append(new_img_id)
            info = self.dset.index.imgs[new_img_id]
            frame_index[1 + j] = info['frame_index']
            sensor_list.append(info['sensor_coarse'])
            date = info['date_captured']
            date_list.append(torch.tensor([int(date[:4]), int(date[5:7])]))
        # load images
        image_dict = dict()
        for k in range(self.num_images):
            image = self.dset.delayed_load(id_list[k], channels=self.channels, space='video').finalize().astype(np.float32)

            image = np.nan_to_num(image)
            if image.std() != 0.:
                image = (image - image.mean()) / image.std()
            image_dict['img{}'.format(1 + k)] = image

        albumentations_input = {'image': image_dict['img1']}
        for k in range(1, self.num_images):
            albumentations_input['image{}'.format(1 + k)] = image_dict['img{}'.format(1 + k)]

        if not self.segmentation_labels:
            # transformations
            transformed = self.transforms(**albumentations_input)
            images = [transformed[key] for key in transformed]

            if self.mode == 'test':
                # additional transforms for test mode
                transformed = self.transforms(image=images[0], image2=images[1])
                transformed2 = self.transforms2(image=images[0])
                img1 = transformed['image']
                img2 = transformed['image2']
                img3 = transformed2['image']
                transformed3 = self.transforms3(image=img1)
                # convert to tensors
                img1 = torch.tensor(img1).permute(2, 0, 1)
                img2 = torch.tensor(img2).permute(2, 0, 1)
                img4 = transformed3['image']
                img3 = torch.tensor(img3).permute(2, 0, 1)
                img4 = torch.tensor(img4).permute(2, 0, 1)

                frame_index1 = self.dset.index.imgs[img1_id]['frame_index']
                frame_index2 = self.dset.index.imgs[id_list[1]]['frame_index']

                out['offset_image1'] = img3.float()
                out['augmented_image1'] = img4.float()
                out['img1_id'] = img1_id
                out['time_sort_label'] = float(frame_index1 < frame_index2)
                # out['img1_info'] = self.dset.index.imgs[img1_id]

            images = [torch.tensor(x).permute(2, 0, 1) for x in images]

            if self.display:
                if self.num_channels == 3:

                    display_image1 = images[0]
                    display_image2 = images[1]
                elif self.num_channels < 10:
                    display_image1 = images[0][[0, 1, 2], :, :]
                    display_image2 = images[1][[0, 1, 2], :, :]
                else:
                    display_image1 = images[0][[3, 2, 1], :, :]
                    display_image2 = images[1][[3, 2, 1], :, :]
                display_image1 = (2 + display_image1) / 6
                display_image2 = (2 + display_image2) / 6
            else:
                display_image1 = torch.tensor([])
                display_image2 = torch.tensor([])

            for m in range(self.num_images):
                out['image{}'.format(1 + m)] = images[m]
                out['frame_number{}'.format(1 + m)] = frame_index[1 + m]

            out['display_image1'] = display_image1
            out['display_image2'] = display_image2

        else:
            aids = [self.dset.index.gid_to_aids[n] for n in id_list]

            dets = [kwimage.Detections.from_coco_annots(self.dset.annots(p).objs, dset=self.dset) for p in aids]

            vid_from_img = [kwimage.Affine.coerce(self.dset.index.imgs[q]['warp_img_to_vid']) for q in id_list]

            dets = [det.warp(img) for (det, img) in zip(dets, vid_from_img)]

            # bbox = dets.data['boxes'].data
            segmentations = []
            category_id = []
            for r in range(self.num_images):
                segmentations.append(dets[r].data['segmentations'].data)
                category_id.append([dets[r].classes.idx_to_id[cidx] for cidx in dets[r].data['class_idxs']])

            img_dims = (image_dict['img1'].shape[0], image_dict['img1'].shape[1])

            masks = dict()

            for s in range(self.num_images):
                combined = []
                for sseg, cid in zip(segmentations[s], category_id[s]):
                    assert cid > 0
                    np_mask = sseg.to_mask(dims=img_dims).data.astype(float) * cid
                    mask = torch.from_numpy(np_mask)
                    combined.append(mask.unsqueeze(0))
                if combined:
                    overall_mask = torch.max(torch.cat(combined, dim=0), dim=0)[0].numpy()  # ### HACK
                else:
                    overall_mask = np.zeros_like(image_dict['img1'][:, :, 0])

                masks['seg{}'.format(1 + s)] = overall_mask

            albumentations_input = {'image': image_dict['img1'], 'mask': masks['seg1']}
            for t in range(1, self.num_images):
                albumentations_input['image{}'.format(1 + t)] = image_dict['img{}'.format(1 + t)]
                albumentations_input['seg{}'.format(1 + t)] = masks['seg{}'.format(1 + t)]

            transformed = self.transforms(**albumentations_input)
            images = [transformed[key] for key in transformed if 'image' in key]

            images = [torch.tensor(x).permute(2, 0, 1) for x in images]
            all_segmentations = [transformed[key] for key in transformed if 'seg' in key or 'mask' in key]
            all_segmentations = [torch.tensor(x) for x in all_segmentations]
            segmentations = [torch.where(seg == 2, 1, 0) + torch.where(seg == 3, 1, 0) + torch.where(seg == 8, 1, 0) - torch.where(seg == 5, 1, 0) - torch.where(seg == 6, 1, 0) for seg in all_segmentations]

            if self.display:
                if self.num_channels == 3:
                    display_image1 = images[0]
                    display_image2 = images[1]
                elif self.num_channels < 10:
                    display_image1 = images[0][[0, 1, 2], :, :]
                    display_image2 = images[1][[0, 1, 2], :, :]
                else:
                    display_image1 = images[0][[3, 2, 1], :, :]
                    display_image2 = images[1][[3, 2, 1], :, :]
                display_image1 = (2 + display_image1) / 6
                display_image2 = (2 + display_image2) / 6
            else:
                display_image1 = torch.tensor([])
                display_image2 = torch.tensor([])

            for m in range(self.num_images):
                out['image{}'.format(1 + m)] = images[m]
                out['frame_number{}'.format(1 + m)] = frame_index[1 + m]
                out['segmentation{}'.format(1 + m)] = segmentations[m]
            out['display_image1'] = display_image1
            out['display_image2'] = display_image2
            out['order'] = [x[0] for x in sorted(frame_index.items(), key=lambda item: item[1])]

        out['frame_number'] = torch.tensor([out[key] for key in out if 'frame_number' in key])
        out['normalized_date'] = torch.tensor([date_[0] - 2018 + date_[1] / 12 for date_ in date_list])
        out['sensors'] = sensor_list
        # out['img1_info'] = img1_info
        out['img1_id'] = img1_id
        return out


class SpaceNet7(Dataset):
    '''TO DO: Return segmentation labels for SpaceNet 7'''
    normalize_params = [[0.16198677, 0.22665408, 0.1745371], [0.06108317, 0.06515977, 0.04128775]]

    def __init__(self,
                    patch_size=128,
                    splits='satellite_sort/data/spacenet/splits_unmasked/',  # ### unmasked images
                    train=True,
                    display=False,
                    num_images=3,
                    segmentation_labels=False):
        self.num_images = num_images
        self.display = display
        self.train = train
        self.segmentation_labels = segmentation_labels

        additional_targets = dict()
        for i in range(1, self.num_images):
            additional_targets['image{}'.format(1 + i)] = 'image'
            additional_targets['mask{}'.format(1 + i)] = 'mask'

        self.transforms = A.Compose([
                #                 A.NoOp(),
                A.RandomCrop(height=patch_size, width=patch_size),
                A.RandomRotate90(p=.999)
                ],
                additional_targets=additional_targets)

        if train:
            self.images = []
            with open(os.path.join(splits, 'spacenet7_train_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))

            self.segmentation_masks = []
            if segmentation_labels:
                with open(os.path.join(splits, 'spacenet7_train_building_masks.txt'), 'r') as file:
                    for row in file:
                        self.segmentation_masks.append(row.strip('\n'))
        else:
            self.images = []
            with open(os.path.join(splits, 'spacenet7_val_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))

            self.segmentation_masks = []
            if segmentation_labels:
                with open(os.path.join(splits, 'spacenet7_val_building_masks.txt'), 'r') as file:
                    for row in file:
                        self.segmentation_masks.append(row.strip('\n'))

        self.length = len(self.images)

    def date_to_step(self, date):
        year, month = date
        return (year % 2018) * 12 + month

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        images = dict()

        im1_path = self.images[idx]
        date1 = (int(im1_path[-47:-43]), int(im1_path[-42:-40]))

        im_paths = [im1_path]
        dates = [date1]

        im_directory, _ = os.path.split(self.images[idx])
        sorted_img_directory = sorted(os.listdir(im_directory))

        date1 = (int(self.images[idx][-47:-43]), int(self.images[idx][-42:-40]))
        image1 = tif.imread(im1_path).astype('float32')
        images['img1'] = image1

        if self.segmentation_labels:
            segmentations = dict()
            mask_directory, _ = os.path.split(self.segmentation_masks[idx])
            sorted_mask_directory = sorted(os.listdir(mask_directory))
            segmentations['seg1'] = tif.imread(self.segmentation_masks[idx]).astype('float32') / 255

        for x in range(1, self.num_images):
            date_new = date1
            while date_new in dates:
                idx_new = torch.randint(len(sorted_img_directory), (1,))[0]

                im_new_path = os.path.join(im_directory, sorted_img_directory[idx_new])
                date_new = (int(im_new_path[-47:-43]), int(im_new_path[-42:-40]))
            if self.segmentation_labels:
                segmentations_new_path = os.path.join(mask_directory, sorted_mask_directory[idx_new])
                segmentations['seg{}'.format(1 + x)] = tif.imread(segmentations_new_path).astype('float32') / 255
            dates.append(date_new)
            im_paths.append(im_new_path)
            image_new = tif.imread(im_new_path).astype('float32')
            images['img{}'.format(1 + x)] = image_new

        albumentations_input = {'image': images['img1']}
        if self.segmentation_labels:
            albumentations_input['mask'] = segmentations['seg1']
        for t in range(1, self.num_images):
            albumentations_input['image{}'.format(1 + t)] = images['img{}'.format(1 + t)]
            if self.segmentation_labels:
                albumentations_input['mask{}'.format(1 + t)] = segmentations['seg{}'.format(1 + t)]

        transformed = self.transforms(**albumentations_input)

        images = [transformed[key][:, :, :3] for key in transformed if 'image' in key]

        normalized_images = [torch.tensor((x - x.mean()) / x.std()).permute(2, 0, 1) for x in images[:self.num_images]]

        cloud_masks = [transformed[key][:, :, 3] / 255 for key in transformed if 'image' in key]

        if self.display:
            display_images = [x.astype('float32') for x in images]
        else:
            display_images = [torch.tensor([]) for x in range(self.num_images)]
        if self.segmentation_labels:
            segmentations = [torch.tensor(transformed[key]) for key in transformed if 'mask' in key]

        out = dict()
        for m in range(self.num_images):
            out['image{}'.format(1 + m)] = normalized_images[m]
            out['date{}'.format(1 + m)] = self.date_to_step(dates[m])
            out['cloud_mask{}'.format(1 + m)] = cloud_masks[m]
            if self.segmentation_labels:
                out['segmentation{}'.format(1 + m)] = segmentations[m]
            if self.display:
                out['display_image{}'.format(1 + m)] = display_images[m]

        sort_order = torch.tensor([y[0] + y[1] / 100 for y in dates[:self.num_images]])
        out['order'] = sort_order.sort()[1]
        out['time_steps'] = torch.tensor([out[key] for key in out if 'date' in key])

        return out
