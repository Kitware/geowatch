import torch
import os
import shutil
import numpy as np

from torch.utils.data import DataLoader
from torchvision import utils
from . import data_config
from .datasets.CD_dataset import CDDataset

# Set up data folder here


def setup_data_folder(data_name,
                      data_split,
                      fname_imgA,
                      fname_imgB,
                      fname_depthA,
                      fname_depthB,
                      is_rectify=False):

    dataConfig = data_config.DataConfig().get_data_config(data_name, is_rectify)

    root = dataConfig.root_dir

    # Make clean folders
    if os.path.exists(root):
        shutil.rmtree(root)

    os.mkdir(root)
    os.mkdir(os.path.join(root, 'A'))
    os.mkdir(os.path.join(root, 'B'))
    os.mkdir(os.path.join(root, 'depth_A'))
    os.mkdir(os.path.join(root, 'depth_B'))
    os.mkdir(os.path.join(root, 'label'))
    os.mkdir(os.path.join(root, 'list'))

    # print('fname_imgA is: ', fname_imgA)
    # print('fname_imgB is: ', fname_imgB)

    # Make soft links
    local_name = fname_imgA.split('/')[-1]
    img_suffix = local_name.split('.')[-1]

    local_nameB = fname_imgB.split('/')[-1][:-4]
    local_name = local_nameB + '-' + local_name

    # set up prefix for original image and depth map
    org_prefix = dataConfig.org_prefix
    depth_prefix = dataConfig.depth_prefix

    if org_prefix not in local_name:
        local_name = org_prefix + local_name

    local_name_depth = local_name.replace(org_prefix, depth_prefix)
    if '.tif' in local_name_depth:
        local_name_depth = local_name_depth.replace('.tif', '.png')

    if is_rectify:
        fname_depthA = fname_depthA.replace('segW_', 'segW1_')
        fname_depthB = fname_depthB.replace('segW_', 'segW1_')
        fname_imgA = fname_depthA.replace('segW1_', 'rect_')
        fname_imgB = fname_depthB.replace('segW1_', 'rect_')
        # rectified images are always tiff files
        if '.tif' not in fname_imgA:
            fname_imgA = fname_imgA.replace(img_suffix, 'tif')
        if '.tif' not in fname_imgB:
            fname_imgB = fname_imgB.replace(img_suffix, 'tif')

    os.symlink(fname_imgA, os.path.join(root, 'A', local_name))
    os.symlink(fname_imgB, os.path.join(root, 'B', local_name))
    os.symlink(fname_imgA, os.path.join(root, 'label', local_name))
    os.symlink(fname_depthA, os.path.join(root, 'depth_A', local_name_depth))
    os.symlink(fname_depthB, os.path.join(root, 'depth_B', local_name_depth))
    # The lable is just a place holder

    fp_list = open(os.path.join(root, 'list', data_split + '.txt'), 'w')
    fp_list.write(local_name)
    fp_list.close()


def get_loader(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, is_rectify=False, dataset='CDDataset'):

    dataConfig = data_config.DataConfig().get_data_config(data_name, is_rectify)

    root_dir = dataConfig.root_dir
    org_prefix = dataConfig.org_prefix
    depth_prefix = dataConfig.depth_prefix
    label_transform = dataConfig.label_transform

    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split, img_size=img_size,
                             is_train=is_train, label_transform=label_transform,
                             org_prefix=org_prefix, depth_prefix=depth_prefix)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return dataloader


def get_loaders(args):

    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'CDDataset':
        training_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=args.img_size, is_train=True,
                                 label_transform=label_transform)
        val_set = CDDataset(root_dir=root_dir, split=split_val,
                                 img_size=args.img_size, is_train=False,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])
