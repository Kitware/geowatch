#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a Template for writing training logic.
"""
#package imports
import torch
from argparse import ArgumentParser, RawTextHelpFormatter
import os
from datetime import date

import torch
import kwcoco
import kwimage
import random
import itertools as it
import tifffile
import numpy as np
from tqdm import tqdm
#local imports
from .model import pretext
from .iarpa_dataset import kwcoco_dataset

def main(args):
    print('Loading checkpoint')
    model = pretext.load_from_checkpoint(args.ckpt_path)
    model.eval().to(args.device)
    print('Initiating dataset')
    dset = kwcoco.CocoDataset.coerce(args.input_kwcoco)
    
    if 'all' in args.bands:
        if args.sensor == 'S2':
            args.bands = ['coastal', 'blue', 'green', 'red', 'B05', 'B06', 'B07', 'nir', 'B09', 'cirrus', 'swir16', 'swir22', 'B8A']
        elif args.sensor == 'L8' or args.sensor == 'LS':
            args.bands = ['coastal', 'lwir11', 'lwir12', 'blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'pan', 'cirrus']
    
    images = dset.images()
    images = images.compress([x == args.sensor for x in images.lookup('sensor_coarse')])
            
    dset_ids = images.gids
    

    if len(args.tasks) == 1 and args.tasks[0].lower() == 'all':
        feature_types = model.TASK_NAMES
        feature_types.append('shared')
    else:
        feature_types = args.tasks
    
    print('Evaluating and saving features')
    
    if args.data_save_folder:
        root = args.data_save_folder
    else:
        root, _ = os.path.split(args.input_kwcoco) 
    
    for img_id in tqdm(dset_ids):
        
        image = dset.index.imgs[img_id]
        
        img1 = dset.delayed_load(img_id, channels = args.bands) 
        img1 = img1.finalize().astype('float32')
        img1 = (img1 - img1.mean())/img1.std()
        img1 = torch.tensor(img1).permute(2,0,1)
        
        aux_base = image['auxiliary'][0]
        path, file_name = os.path.split(aux_base['file_name'])
        
        if args.data_save_folder:
            save_path = args.data_save_folder
        else:
            save_path = os.path.join(root, 'uky_invariants', path)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        features = model.predict(img1.unsqueeze(0).to(args.device))
        
        for key in feature_types:
            
            feat = features[key].squeeze()
            feat = feat.permute(1,2,0).detach().cpu().numpy()
            last_us_idx = file_name.rfind('_')
            name = file_name[:last_us_idx] + '_invariants_' + key + '.tif'
            tifffile.imsave(os.path.join(save_path, name), feat)
            
            warp_img_to_vid = kwimage.Affine.coerce(image.get('warp_img_to_vid', None))
            warp_aux_to_img = warp_img_to_vid.inv()
            
            info = {}
            info['file_name'] = os.path.join('uky_invariants', path, name)
            info['height'] = feat.shape[0]
            info['width'] = feat.shape[1]
            info['num_bands'] = feat.shape[2]
            info['channels'] = '|'.join(['inv_' + key + f'{i}' for i in range(1, feat.shape[2]+1)])
            info['warp_aux_to_img'] = kwimage.Affine.coerce(warp_aux_to_img).concise()
            
#             for tmp in dset.index.imgs[img_id]['auxiliary']:
#                 print(tmp['file_name'])
#             1/0
            
            dset.index.imgs[img_id]['auxiliary'].append(info)
    
    if not args.output_kwcoco:
        args.output_kwcoco = args.input_kwcoco

    dset.fpath = args.output_kwcoco
    print('Write to dset.fpath = {!r}'.format(dset.fpath))
    dset.dump(dset.fpath, newlines=True)
    
    print('Done')

if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.tasks.template.fit --help

        python -m watch.tasks.template.fit \
            --train_dataset=special:vidshapes32-multispectral \
            --vali_dataset=special:vidshapes16-multispectral
    """
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--device', type=str, default='cuda')
    
    ####pytorch lightning checkpoint
    parser.add_argument('--ckpt_path', type=str, required=True)
    
    ####data flags - make sure these match the trained checkpoint
    parser.add_argument('--sensor', type=str, help='Sensor to generate features from. Currently must choose from S2, LS, or L8. Make sure this matches the sensor used to train the model in ckpt_path.', default='S2')
    parser.add_argument('--bands', type=list, help='Bands to use in evaluation. Choose from \'all\' or create list of acceptable bands. Make sure this matches the bands used to train the model in ckpt_path.', default=['all'])
    
    ####output flags
    parser.add_argument('--data_save_folder', help='Path to store generated feature tifs, data will end up within a folder named \'uky_invariants\'. If not specified, data is saved in the folder uky_invariants within the overall data folder.', type=str)
    parser.add_argument('--input_kwcoco', type=str, help='Path to kwcoco dataset with images to generate feature for', required=True)
    parser.add_argument('--output_kwcoco', type=str, help='Path to write an outut kwcoco file. Output file will be a copy of input_kwcoco with addition feature fields generated by predict.py. If None, output_kwcoco will update input kwcoco.', default='')
    parser.add_argument('--tasks', nargs='+', help=f'specify which tasks to choose from ({", ".join(pretext.TASK_NAMES)}, shared, or all.\nEx: --tasks {pretext.TASK_NAMES[0]} {pretext.TASK_NAMES[1]}', default=['all'])
    
    parser.set_defaults(
        terminate_on_nan=True
        )
   
    args = parser.parse_args()

    #check save directory
    default_path = os.path.join(os.getcwd(), 'watch/tasks/invariants')
    if args.ckpt_path == None and os.path.exists(default_path):
        args.ckpt_path = os.path.join(default_path, 'logs')
    elif args.ckpt_path == None:
        args.ckpt_path = 'invariants_logs'
    
    torch.autograd.set_detect_anomaly(True)
    main(args)
    
    