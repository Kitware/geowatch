import torch
from torch.utils import data
import os
import os.path as osp


def main(data_dir):
    data_folder = os.listdir(data_dir)

    train_folders, val_folders = data.random_split(
        data_folder, [50, 10], generator=torch.Generator().manual_seed(12))

    train_images = []
    train_building_masks = []
    for x in train_folders:
        for y in os.listdir(osp.join(data_dir, x, 'images_masked')):
            train_images.append(osp.join(data_dir, x, 'images_masked', y))
        for z in os.listdir(osp.join(data_dir, x, 'building_masks')):
            train_building_masks.append(osp.join(data_dir, x, 'building_masks', z))

    train_images = sorted(train_images)
    train_building_masks = sorted(train_building_masks)

    val_images = []
    val_building_masks = []
    for x in val_folders:
        for y in os.listdir(osp.join(data_dir, x, 'images_masked')):
            val_images.append(osp.join(data_dir, x, 'images_masked', y))
        for z in os.listdir(osp.join(data_dir, x, 'building_masks')):
            val_building_masks.append(osp.join(data_dir, x, 'building_masks', z))

    val_images = sorted(val_images)
    val_building_masks = sorted(val_building_masks)

    with open('./spacenet/data/spacenet7_train_images.txt', 'w') as file:
        for item in train_images:
            file.write('%s\n' % item)

    with open('./spacenet/data/spacenet7_train_building_masks.txt', 'w') as file:
        for item in train_building_masks:
            file.write('%s\n' % item)

    with open('./spacenet/data/spacenet7_val_images.txt', 'w') as file:
        for item in val_images:
            file.write('%s\n' % item)

    with open('./spacenet/data/spacenet7_val_building_masks.txt', 'w') as file:
        for item in val_building_masks:
            file.write('%s\n' % item)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/localdisk0/SCRATCH/watch/SpaceNet/7/train/')
    args = parser.parse_args()

    main(args.data_dir)
