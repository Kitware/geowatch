import tifffile as tif
import torch
from torch.utils import data
import os
import os.path as osp

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/localdisk0/SCRATCH/watch/SpaceNet/7/train/')
    parser.add_argument('--num_unlabeled', type=int, default=45)
    parser.add_argument('--num_labeled_train', type=int, default=10)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    data_folder = os.listdir(args.data_dir)
    
    unlabeled_folders, train_folders, val_folders = data.random_split(data_folder, [args.num_unlabeled, args.num_labeled_train, 60-args.num_unlabeled-args.num_labeled_train], generator=torch.Generator().manual_seed(38))

    unlabeled_images = []
    unlabeled_building_masks = []
    for x in unlabeled_folders:
        for y in os.listdir(osp.join(data_dir, x, 'images_masked')):
            unlabeled_images.append(osp.join(data_dir, x, 'images_masked', y))
        for z in os.listdir(osp.join(data_dir, x, 'building_masks')):
            unlabeled_building_masks.append(osp.join(data_dir, x, 'building_masks', z))

    unlabeled_images = sorted(unlabeled_images)
    unlabeled_building_masks = sorted(unlabeled_building_masks)
 
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
    
    with open('./spacenet7_unlabeled_images.txt', 'w') as file:
        for item in unlabeled_images:
            file.write('%s\n' % item)

    with open('./spacenet7_unlabeled_building_masks.txt', 'w') as file:
        for item in unlabeled_building_masks:
            file.write('%s\n' % item)

    with open('./spacenet7_train_images.txt', 'w') as file:
        for item in train_images:
            file.write('%s\n' % item)
        
    with open('./spacenet7_train_building_masks.txt', 'w') as file:
        for item in train_building_masks:
            file.write('%s\n' % item)

    with open('./spacenet7_val_images.txt', 'w') as file:
        for item in val_images:
            file.write('%s\n' % item)

    with open('./spacenet7_val_building_masks.txt', 'w') as file:
        for item in val_building_masks:
            file.write('%s\n' % item)
