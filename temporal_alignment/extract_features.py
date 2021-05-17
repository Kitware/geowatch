import torch
import kwcoco
import os
import json 
import tifffile

from time_sort_drop0 import time_sort
from drop0_datasets import drop0_pairs

def extract_features(checkpoint,
                    data_folder,
                    kwcoco_file,
                    output_kwcoco,
                    output_folder,
                    image_ids,
                    sensor, 
                    panchromatic,
                    device
                    ):

    extractor = time_sort.load_from_checkpoint(checkpoint, map_location='cuda').to(device)
    
#     dataset = kwcoco.CocoDataset(kwcoco)
    with open(kwcoco_file) as read:
        dataset = json.load(read)
    
    if not os.path.exists(output_kwcoco):
        with open(output_kwcoco, 'w') as new_file:
            json.dump(dataset, new_file)

    dataset = kwcoco.CocoDataset(output_kwcoco)

    if not image_ids:
        image_ids = range(1,1+len(dataset.imgs))
    
    sensor_list = dataset.images().lookup('sensor_coarse', keepid=True)
    sensor_ids = [ID for ID in sensor_list if sensor_list[ID] == sensor]
    
    for x in image_ids:
        if x in sensor_ids:
            print('Processing image {}'.format(x))
            file_name=dataset.imgs[x]['file_name']
            
            directory, _ = os.path.split(file_name)

            if not os.path.exists(os.path.join(output_folder,directory)):
                os.makedirs(os.path.join(output_folder,directory))
            
            image = torch.tensor(tifffile.imread(os.path.join(data_folder, file_name)).astype('int32')).to(device).permute(2,0,1).float()
            while len(image.shape) < 4:
                image = image.unsqueeze(0)
            features, _, _, _ = extractor(image, image, 'x', 'x')
            save_name = file_name[:-4] + '.pt' 
            torch.save(features.squeeze(), os.path.join(output_folder, save_name))
            dataset.imgs[x]['features'] = os.path.join(output_folder, save_name)
            with open(dataset.fpath) as new_file:
                dataset.dump(dataset.fpath, newlines=True)
        else:
            print('Skipping image {}, sensor doesn\'t match'.format(x))
    


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', default='logs/drop0_sort/train_vidio_1/2021-05-14/default/version_7/checkpoints/epoch=97-step=195.ckpt') ###change to some shared place
    
    ### drop0_aligned dataset arguments
    parser.add_argument('--panchromatic', help='set flag for using panchromatic landsat imagery', action='store_true')
    parser.add_argument('--sensor', type=str, help='Choose from WV, LC, or S2', default='S2')   
    parser.add_argument('--data_folder', default='/u/eag-d1/data/watch/drop0_aligned/')
    parser.add_argument('--kwcoco_file', help='kwcoco file with dataset', default='/u/eag-d1/data/watch/drop0_aligned/data.kwcoco.json')
    parser.add_argument('--output_kwcoco', help='Filename to save output kwcoco', default='/u/eag-d1/scratch/ben/test.kwcoco.json')
    parser.add_argument('--output_folder', help='Folder to store output feature tenors as .pt files', default='/u/eag-d1/scratch/ben/drop0_features/')
    parser.add_argument('--image_ids', nargs='+', type=int, help='Set to 0 for all available images. Otherwise input list of image ids for processing.', default=0)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
 
    extract_features(checkpoint=args.checkpoint,
                    data_folder=args.data_folder,
                    kwcoco_file=args.kwcoco_file,
                    output_kwcoco=args.output_kwcoco,
                    output_folder=args.output_folder,
                    image_ids=args.image_ids,
                    sensor=args.sensor, 
                    panchromatic=args.panchromatic,
                    device=args.device
                    )

#     if not args.image_ids:
#         args.image_ids = range(1,389)
    
#     extractor = time_sort.load_from_checkpoint(args.checkpoint, map_location='cuda').to(device)
    
# #     dataset = kwcoco.CocoDataset(args.kwcoco)
#     with open(args.kwcoco) as read:
#         dataset = json.load(read)
    
#     if not os.path.exists(args.output_kwcoco):
#         with open(args.output_kwcoco, 'w') as new_file:
#             json.dump(dataset, new_file)
#     print(args.output_kwcoco)
#     dataset = kwcoco.CocoDataset(args.output_kwcoco)
    
#     sensor_list = dataset.images().lookup('sensor_coarse', keepid=True)
#     sensor_ids = [ID for ID in sensor_list if sensor_list[ID] == args.sensor]
    
#     for x in args.image_ids:
#         if x in sensor_ids:
#             print('Processing image {}'.format(x))
#             file_name=dataset.imgs[x]['file_name']
            
#             directory, _ = os.path.split(file_name)

#             if not os.path.exists(os.path.join(args.output_folder,directory)):
#                 os.makedirs(os.path.join(args.output_folder,directory))
            
#             image = torch.tensor(tifffile.imread(os.path.join(args.root, file_name)).astype('int32')).to(device).permute(2,0,1).float()
#             while len(image.shape) < 4:
#                 image = image.unsqueeze(0)
#             features, _, _, _ = extractor(image, image, 'x', 'x')
#             save_name = file_name[:-4] + '.pt' 
#             torch.save(features.squeeze(), os.path.join(args.output_folder, save_name))
#             dataset.imgs[x]['features'] = os.path.join(args.output_folder, save_name)
#             with open(args.output_kwcoco) as new_file:
#                 dataset.dump(dataset.fpath, newlines=True)
#         else:
#             print('Skipping image {}, sensor doesn\'t match'.format(x))
    
        
    
    