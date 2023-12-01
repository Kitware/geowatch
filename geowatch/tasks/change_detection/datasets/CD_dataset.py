"""
变化检测数据集
"""

import os
import cv2
import torch
import numpy as np

from PIL import Image
from pytorch_msssim import ssim
from torch.utils import data
from .data_utils import CDDataAugmentation
from scipy import ndimage
from skimage import feature
from skimage.registration import phase_cross_correlation

from ..dzyne_img_util import load_image, normalizeRGB, tensor2np
from ..dzyne_align_util import optical_flow, warp_flow, featureAlign

"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
├─depth
├─depth_post
└─list
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"
DEPTH_FOLDER_NAME = "depth_A"
DEPTH_POST_FOLDER_NAME = 'depth_B'

IGNORE = 255

label_suffix = '.png'  # jpg for gan dataset, others : png
depth_suffix = '.png'


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)

    if img_name_list.ndim == 2:
        return img_name_list[:, 0]

    if img_name_list.ndim == 0:
        img_name_list = np.expand_dims(img_name_list, 0)

    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_post_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))


def get_depth_path(root_dir, img_name, org_prefix, depth_prefix):
    img_name = img_name.replace(org_prefix, '')

    return os.path.join(root_dir, DEPTH_FOLDER_NAME,
                        depth_prefix + img_name.replace('.jpg', depth_suffix))


def get_depth_post_path(root_dir, img_name, org_prefix, depth_prefix):
    img_name = img_name.replace(org_prefix, '')

    return os.path.join(root_dir, DEPTH_POST_FOLDER_NAME,
                        depth_prefix + img_name.replace('.jpg', depth_suffix))

#--------------------------------------------------------
# Compute structural similiarity index SSIM
#--------------------------------------------------------


def compute_ssim(img, img_B):
    height, width, channels = img_B.shape

    timg = torch.Tensor(np.expand_dims(np.swapaxes(np.swapaxes(img, 0, 2), 1, 2), axis=0))
    timg_B = torch.Tensor(np.expand_dims(np.swapaxes(np.swapaxes(img_B, 0, 2), 1, 2), axis=0))

    ssim_val, _, ssim_map = ssim(timg, timg_B, data_range=255, size_average=False, full=True)

    tmp0 = tensor2np(torch.squeeze(ssim_map))

    offset0 = int((height - tmp0.shape[0]) / 2)
    offset1 = int((width - tmp0.shape[1]) / 2)

    big_ssim_map = np.zeros([height, width])
    big_ssim_map[offset0:tmp0.shape[0] + offset0, offset1:tmp0.shape[1] + offset1] = ssim_map[0, 1, :, :]

    # tmp0 = tensor2np(torch.squeeze(ssim_map))
    Image.fromarray((255 * big_ssim_map).astype(np.uint8)).save('/media/FastData/yguo/ssim.png')

    return big_ssim_map
#=================================================================


class ImageDataset(data.Dataset):
    """VOCdataloder"""

    def __init__(self, root_dir, split='train', img_size=256, is_train=True, to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        # self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.list + '.txt'
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')

        self.img_name_list = load_img_name_list(self.list_path)

        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        # Align the two images if needed
        translation = feature.register_translation(img, img_B, upsample_factor=10)[0]
        img_B = ndimage.shift(img_B, translation)

        [img, img_B], _ = self.augm.transform([img, img_B], [], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
                 org_prefix='org_', depth_prefix='segW_',
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform
        self.org_prefix = org_prefix
        self.depth_prefix = depth_prefix

    def __getitem__(self, index):
        name = self.img_name_list[index]
        # print('root_dir is:', self.root_dir)

        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        # nameB = B_path.split('/')[-1].split('.')[0]
        # name = nameB + '-' + name

        # img = np.asarray(Image.open(A_path).convert('RGB'))
        # img_B = np.asarray(Image.open(B_path).convert('RGB'))
        img = normalizeRGB(load_image(A_path, A_path))
        img_B = normalizeRGB(load_image(B_path, B_path))
        img = (255 * img).astype(np.uint8)
        img_B = (255 * img_B).astype(np.uint8)

        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])
        # label = np.array(Image.open(L_path), dtype=np.uint8)
        label = (255 * load_image(L_path, L_path)).astype(np.uint8)

        # print('self org_prefix & depth_prefix are: ', self.org_prefix, self.depth_prefix)

        depth_A_path = get_depth_path(self.root_dir, self.img_name_list[index % self.A_size],
                                      self.org_prefix, self.depth_prefix)
        depth_B_path = get_depth_post_path(self.root_dir, self.img_name_list[index % self.A_size],
                                           self.org_prefix, self.depth_prefix)

        if '.tif' in depth_A_path:
            depth_A_path = depth_A_path.replace('.tif', '.png')

        if '.tif' in depth_B_path:
            depth_B_path = depth_B_path.replace('.tif', '.png')

        # print('depth A path is:', depth_A_path)
        # print('depth B path is:', depth_B_path)

        depth = np.asarray(Image.open(depth_A_path).convert('P'))
        depth_B = np.asarray(Image.open(depth_B_path).convert('P'))

        # print('min/max image_B are:', img_B.min(), img_B.max())
        # print('min/max depth_B are:', depth_B.min(), depth_B.max())

        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label = label // 255

        height, width, _ = img_B.shape

        # resize images to be the same size
        height_ref, width_ref, _ = img.shape

        if height != height_ref or width != width_ref:
            new_width = np.minimum(width, width_ref)
            new_height = np.minimum(height, height_ref)
            new_size = (new_width, new_height)

            img = cv2.resize(img, new_size)
            img_B = cv2.resize(img_B, new_size)
            depth = cv2.resize(depth, new_size)
            depth_B = cv2.resize(depth_B, new_size)
            label = cv2.resize(label, new_size)

        if 0:
            img_B, transform = featureAlign(img_B, img)
            depth_B = cv2.warpPerspective(depth_B, transform, (width, height))

        else:
            # Align the two images if needed
            translation = phase_cross_correlation(img, img_B, upsample_factor=10)[0]  # skimage.feature.registration_translation got deprecated

            # disable the shift in Z direction
            translation[2] = 0

            # print('max is: ', abs(translation).max())
            if abs(translation).max() > 30:
                translation = 0

            img_B = ndimage.shift(img_B, translation)
            # print('translation is :', translation)

            # depth_B = ndimage.shift(np.expand_dims(depth_B, axis=2), translation)[:,:,0]
            depth_B = ndimage.shift(np.expand_dims(depth_B, axis=2), translation)
            # print('min/max image_B after shift are:', img_B.min(), img_B.max())

            flow, flow_norm = optical_flow(img, img_B)
            # print('min/max flow are:', flow.min(), flow.max())

            img_B = warp_flow(img_B, flow)
            depth_B = warp_flow(depth_B, flow)

        # print('img imgB shapes are:', img.shape, img_B.shape)
        # corr = compute_ssim(img, img_B)
        corr = np.ones((img_B.shape[0], img_B.shape[1]))  # TODO - compute_ssim needs Yanlin's mods to work

        [img, img_B], [depth, depth_B], [label] = \
            self.augm.transform([img, img_B], [depth, depth_B], [label], to_tensor=self.to_tensor)

        # print('min/max depth_B after transform are:', depth_B.min(), depth_B.max())

        # print(label.max())
        return {'name': name,
                'A': img, 'B': img_B,
                'depth_A': depth, 'depth_B': depth_B,
                'L': label,
                'corr': corr}
