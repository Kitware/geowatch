import tifffile as tif
import torch
from torch.utils import data
import albumentations as A
import os
import os.path as osp


class S7_sort(data.Dataset):
    normalize_params = [
        [0.16198677, 0.22665408, 0.1745371],
        [0.06108317, 0.06515977, 0.04128775]
    ]

    def __init__(self,
                 crop_size=[128, 128],
                 splits='./spacenet/data/splits_unmasked/',  # unmasked images
                 train=True,
                 normalize=False,
                 num_images=2,
                 display=False
                 ):
        self.display = display
        self.train = train
        self.crop = A.Compose([A.RandomCrop(
            height=crop_size[0], width=crop_size[1])], additional_targets={'image2': 'image'})
        if self.train:
            self.rotate = A.Compose(
                [A.RandomRotate90()], additional_targets={'image2': 'image'})
            self.transforms = A.Compose([
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.),
            ],
                additional_targets={'image2': 'image'}
            )
        else:
            self.transforms = None

        if train:
            self.images = []
            with open(osp.join(splits, 'spacenet7_train_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))

        else:
            self.images = []
            with open(osp.join(splits, 'spacenet7_val_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))

        self.length = len(self.images)
        self.normalize = normalize
        self.normalization = A.Normalize(
            mean=self.normalize_params[0],
            std=self.normalize_params[1],
            max_pixel_value=1.0)
        self.num_images = num_images

    def __len__(self):
        return (self.length)

    def __getitem__(self, idx):

        im_directory, _ = os.path.split(self.images[idx])

        date1 = (int(self.images[idx][-47:-43]),
                 int(self.images[idx][-42:-40]))

        date2 = date1
        while date2 == date1:
            idx2 = torch.randint(len(os.listdir(im_directory)), (1,))[0]
            im2_path = os.path.join(
                im_directory, sorted(
                    os.listdir(im_directory))[idx2])
            date2 = (int(im2_path[-47:-43]), int(im2_path[-42:-40]))

        image = (tif.imread(self.images[idx])).astype(
            "float32")[:, :, :3] / 255
        image2 = (tif.imread(im2_path)).astype("float32")[:, :, :3] / 255

        crop = self.crop(image=image, image2=image2)

        image = crop['image']
        image2 = crop['image2']

        if self.display:
            image_orig, image2_orig = image, image2

        if self.normalize:
            image = self.normalization(image=image)['image']

        if self.transforms:
            image = self.transforms(image=image)['image']
            image2 = self.transforms(image=image2)['image']

            rotated = self.rotate(image=image, image2=image2)
            image = rotated['image']
            image2 = rotated['image2']

        if self.display:
            return torch.tensor(image).permute(2, 0, 1), torch.tensor(image2).permute(
                2, 0, 1), date1, date2, torch.tensor(image_orig).permute(2, 0, 1), torch.tensor(image2_orig).permute(2, 0, 1)

        item = {
            'image1': torch.tensor(image).permute(2, 0, 1),
            'image2': torch.tensor(image2).permute(2, 0, 1),
            'date1': date1,
            'date2': date2
        }
        return item


class S7_time_aligned(data.Dataset):
    normalize_params = [
        [0.16198677, 0.22665408, 0.1745371],
        [0.06108317, 0.06515977, 0.04128775]
    ]

    def __init__(self,
                 crop_size=[128, 128],
                 splits='./spacenet/data/splits_unmasked/',  # unmasked images
                 train=True,
                 normalize=False
                 ):

        self.train = train
        self.crop = A.Compose([A.RandomCrop(
            height=crop_size[0], width=crop_size[1])], additional_targets={'image2': 'image'})
        if self.train:
            self.rotate = A.Compose(
                [A.RandomRotate90()], additional_targets={'image2': 'image'})
            self.transforms = A.Compose([
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.),
            ],
                additional_targets={'image2': 'image'}
            )
        else:
            self.transforms = None

        if train:
            self.images = []
            with open(osp.join(splits, 'spacenet7_train_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))

        else:
            self.images = []
            with open(osp.join(splits, 'spacenet7_val_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))

        self.length = len(self.images)
        self.normalize = normalize
        self.normalization = A.Normalize(
            mean=self.normalize_params[0],
            std=self.normalize_params[1],
            max_pixel_value=1.0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        im_directory, _ = os.path.split(self.images[idx])

        idx2 = idx
        while abs(idx2 - idx) < 1:
            idx2 = torch.randint(len(os.listdir(im_directory)), (1,))[0]
        im2_path = os.path.join(
            im_directory, sorted(
                os.listdir(im_directory))[idx2])

        image = (tif.imread(self.images[idx])).astype(
            "float32")[:, :, :3] / 255
        image2 = (tif.imread(im2_path)).astype("float32")[:, :, :3] / 255

        idx_false = idx
        im_false_directory, _ = os.path.split(self.images[idx_false])

        while im_false_directory == im_directory:
            idx_false = torch.randint(self.length, (1,))[0]
            im_false_directory, _ = os.path.split(self.images[idx_false])

        image_false = (tif.imread(self.images[idx2])).astype(
            "float32")[:, :, :3] / 255

        crop = self.crop(image=image, image2=image2)
        crop_false = self.crop(image=image_false)

        image = crop['image']
        image2 = crop['image2']
        image_false = crop_false['image']

        if self.normalize:
            image = self.normalization(image=image)['image']
            image2 = self.normalization(image=image2)['image']
            image_false = self.normalization(image=image_false)['image']

        if self.transforms:
            image = self.transforms(image=image)['image']
            image_false = self.transforms(image=image_false)['image']
            image2 = self.transforms(image=image2)['image']

            rotated = self.rotate(image=image, image2=image2)
            image = rotated['image']
            image2 = rotated['image2']

        date1 = (int(self.images[idx][-47:-43]),
                 int(self.images[idx][-42:-40]))
        date2 = (int(im2_path[-47:-43]), int(im2_path[-42:-40]))

        item = {
            'image1': torch.tensor(image).permute(2, 0, 1),
            'image2': torch.tensor(image2).permute(2, 0, 1),
            'date1': date1,
            'date2': date2
        }
        return item


class S7_change_augmentations(data.Dataset):
    normalize_params = [
        [0.16198677, 0.22665408, 0.1745371],
        [0.06108317, 0.06515977, 0.04128775]]  # fix

    def __init__(
        self,
        splits='./spacenet/data/',
        crop_size=[128, 128],
        normalize=True,
        train=True
    ):

        self.train = train
        self.crop = A.Compose([
            A.RandomCrop(height=crop_size[0], width=crop_size[1])],
            additional_targets={
                'image2': 'image', 'mask2': 'mask'})
        if self.train:
            self.transforms = A.Compose([
                A.RandomRotate90(),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.),
            ], additional_targets={
                'image2': 'image', 'mask2': 'mask'})
        else:
            self.transforms = None

        self.normalize = normalize
        self.normalization = A.Normalize(
            mean=self.normalize_params[0],
            std=self.normalize_params[1],
            max_pixel_value=1.0)

        if train:
            self.images = []
            with open(osp.join(splits, 'spacenet7_train_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))
            self.masks = []
            with open(osp.join(splits, 'spacenet7_train_building_masks.txt'), 'r') as file:
                for row in file:
                    self.masks.append(row.strip('\n'))
        else:
            self.images = []
            with open(osp.join(splits, 'spacenet7_val_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))
            self.masks = []
            with open(osp.join(splits, 'spacenet7_val_building_masks.txt'), 'r') as file:
                for row in file:
                    self.masks.append(row.strip('\n'))

        self.num_images = len(self.images)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        #idx = idx % self.num_images
        im_directory, _ = os.path.split(self.images[idx])
        mask_directory, _ = os.path.split(self.masks[idx])

        idx2 = idx
        while abs(idx2 - idx) < 5:
            idx2 = torch.randint(len(os.listdir(im_directory)), (1,))[0]
        im2_path = os.path.join(
            im_directory, sorted(
                os.listdir(im_directory))[idx2])
        mask2_path = os.path.join(
            mask_directory, sorted(
                os.listdir(mask_directory))[idx2])

        image = (tif.imread(self.images[idx])).astype("float32") / 255
        image2 = (tif.imread(im2_path)).astype("float32") / 255
        # buildings are encoded as 255, bring this down to 1
        buildings = tif.imread(self.masks[idx]).astype("int32") // 255
        buildings2 = tif.imread(mask2_path).astype("int32") // 255

        crop = self.crop(image=image, mask=buildings, image2=image2,
                         mask2=buildings2)

        image = crop['image']
        image2 = crop['image2']
        buildings = crop['mask']
        buildings2 = crop['mask2']

        if self.normalize:
            image = self.normalization(image=image)['image']
            image2 = self.normalization(image=image2)['image']

        if self.transforms:
            transformed = self.transforms(
                image=image[:, :, 0:3], image2=image2[:, :, 0:3],
                mask=buildings, mask2=buildings2)
            image[:, :, 0:3] = transformed["image"]
            buildings = transformed["mask"]
            image2[:, :, 0:3] = transformed["image2"]
            buildings2 = transformed["mask2"]

        date1 = (int(self.masks[idx][-47:-43]), int(self.masks[idx][-42:-40]))
        date2 = (int(mask2_path[-47:-43]), int(mask2_path[-42:-40]))

        assert (mask2_path[-47:-43] + '-' + mask2_path[-42:-40] ==
                im2_path[-47:-43] + '-' + im2_path[-42:-40])

        if date2 < date1:
            date1, date2 = date2, date1
            image, image2 = image2, image
            buildings, buildings2 = buildings2, buildings

        region = self.images[idx][-32:-4]

        item = {
            'image1': torch.tensor(image).permute(2, 0, 1),
            'image2': torch.tensor(image2).permute(2, 0, 1),
            'label1': torch.tensor(buildings),
            'label2': torch.tensor(buildings2),
            'date1': date1,
            'date2': date2,
            'region': region
        }
        return item


class S7_change(data.Dataset):
    normalize_params = [
        [0.16198677, 0.22665408, 0.1745371],
        [0.06108317, 0.06515977, 0.04128775]
    ]  # fix

    def __init__(
        self,
        splits='./spacenet/data/',
        crop_size=[128, 128],
        normalize=True,
        transforms=None,
        train=True
    ):

        self.crop = A.Compose([
            A.RandomCrop(
                height=crop_size[0],
                width=crop_size[1])],
            additional_targets={
                'image2': 'image',
                'mask2': 'mask'})
        if transforms:
            self.transforms = A.Compose(transforms)

        self.normalize = normalize
        self.normalization = A.Normalize(
            mean=self.normalize_params[0],
            std=self.normalize_params[1],
            max_pixel_value=1.0)

        if train:
            self.images = []
            with open(osp.join(splits, 'spacenet7_train_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))
            self.masks = []
            with open(osp.join(splits, 'spacenet7_train_building_masks.txt'), 'r') as file:
                for row in file:
                    self.masks.append(row.strip('\n'))
        else:
            self.images = []
            with open(osp.join(splits, 'spacenet7_val_images.txt'), 'r') as file:
                for row in file:
                    self.images.append(row.strip('\n'))
            self.masks = []
            with open(osp.join(splits, 'spacenet7_val_building_masks.txt'), 'r') as file:
                for row in file:
                    self.masks.append(row.strip('\n'))

        self.num_images = len(self.images)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        #idx = idx % self.num_images
        im_directory, _ = os.path.split(self.images[idx])
        mask_directory, _ = os.path.split(self.masks[idx])

        idx2 = idx
        while abs(idx2 - idx) < 5:
            idx2 = torch.randint(len(os.listdir(im_directory)), (1,))[0]
        im2_path = os.path.join(
            im_directory, sorted(
                os.listdir(im_directory))[idx2])
        mask2_path = os.path.join(
            mask_directory, sorted(
                os.listdir(mask_directory))[idx2])

        image = (tif.imread(self.images[idx])).astype("float32") / 255
        image2 = (tif.imread(im2_path)).astype("float32") / 255
        # buildings are encoded as 255, bring this down to 1
        buildings = tif.imread(self.masks[idx]).astype("int32") // 255
        buildings2 = tif.imread(mask2_path).astype("int32") // 255

        crop = self.crop(
            image=image,
            mask=buildings,
            image2=image2,
            mask2=buildings2)

        image = crop['image']
        image2 = crop['image2']
        buildings = crop['mask']
        buildings2 = crop['mask2']

        if self.normalize:
            image = self.normalization(image=image)['image']
            image2 = self.normalization(image=image2)['image']

        # if self.transforms:
        #     transformed = self.transforms(image=image, mask=buildings)
        #     image = transformed["image"]
        #     buildings = transformed["mask"]

        date1 = (int(self.masks[idx][-47:-43]), int(self.masks[idx][-42:-40]))
        date2 = (int(mask2_path[-47:-43]), int(mask2_path[-42:-40]))

        assert (mask2_path[-47:-43] + '-' + mask2_path[-42:-40] ==
                im2_path[-47:-43] + '-' + im2_path[-42:-40])

        if date2 < date1:
            date1, date2 = date2, date1
            image, image2 = image2, image
            buildings, buildings2 = buildings2, buildings

        region = self.images[idx][-32:-4]

        item = {
            'image1': torch.tensor(image).permute(2, 0, 1),
            'image2': torch.tensor(image2).permute(2, 0, 1),
            'label1': torch.tensor(buildings),
            'label2': torch.tensor(buildings2),
            'date1': date1,
            'date2': date2,
            'region': region
        }
        return item
