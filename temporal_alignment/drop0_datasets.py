import torch
import kwcoco
import torch.nn.functional as F
import PIL
from PIL import Image
import torchvision.transforms as transforms
import tifffile
import rasterio
import os.path as osp
import random

import numpy as np
from matplotlib.path import Path

class drop0_pairs(torch.utils.data.Dataset):
    def __init__(self, sensor='S2', sites='all', panchromatic=True, video=1, soften_by=0, min_time_step=1, change_labels=list(range(14))):
        
        self.dataset = drop0_aligned_segmented(sensor=sensor, sites=sites, panchromatic=panchromatic, video=video, change_labels=change_labels)
        self.soften_by = soften_by
        self.length = len(self.dataset)
        self.min_time_step=min_time_step
    
    def __len__(self,):
        return self.length
    
    def __getitem__(self, idx):
        im, seg, date, _ = self.dataset.__getitem__(idx)
        idx2 = idx
        while abs(idx2 - idx) < self.min_time_step:
            idx2 = random.randint(0,self.length-1)
        im2, _, date2, _ = self.dataset.__getitem__(idx2)
        
        date = (int(date[:4]),int(date[5:7]),int(date[8:]))
        date2 = (int(date2[:4]),int(date2[5:7]),int(date2[8:]))
        
        if date2 < date:
            im, im2 = im2, im
        
        date = torch.tensor(date)
        date2 = torch.tensor(date2)
        
        return im, im2, date, date2



class drop0_aligned_change(torch.utils.data.Dataset):
    def __init__(self, sensor='S2', sites='all', panchromatic=True, video=1, soften_by=0, change_labels=list(range(14))):
        self.dataset = drop0_aligned_segmented(sensor=sensor, sites=sites, panchromatic=panchromatic, video=video, change_labels=change_labels)
        self.soften_by = soften_by
        self.length = len(self.dataset)
    def __len__(self,):
        return self.length
    
    def __getitem__(self, idx):
        im, seg, date, _ = self.dataset.__getitem__(idx)
        idx2 = idx
        while abs(idx2 - idx) < 1:
            idx2 = random.randint(0,self.length-1)
        im2, seg2, date2, _ = self.dataset.__getitem__(idx2)
        
        if date2 < date:
            im, im2 = im2, im
            seg, seg2 = seg2, seg
        #    date, date2 = date2, date
            
        cmap = torch.where(seg2 - seg != 0, 1., 0. + self.soften_by)
        
        return im, im2, cmap

class drop0_aligned_segmented(torch.utils.data.Dataset):
    """Dataset compatible with drop0_aligned_v2 (now just drop0_aligned on DVC). Sites must be a list from the following: 
                    ['AE-Dubai-0001',
                     'BR-Rio-0270',
                     'BR-Rio-0277',
                     'KR-Pyeongchang-S2',
                     'KR-Pyeongchang-WV',
                     'US-Waynesboro-0001']
    or be set as 'all'.
    
    Sensor must be 'WV' (Worldview), 'LC' (LandCover) or 'S2' (Sentinel 2). If 'WV' is chosen, specify whether you want panchromatic (single channel) images, by setting panchromatic=True. If False, 8 channel multi-spectral images will be returned.
    
    In current drop, all Sentinel 2 images are RGB only. Annotations give bounding box/segmentation outlines of construction sites, but we do not have pixel level annotations for building segmentation or change detection.
    
    There are 5 "videos" in the dataset of aligned images across a single location. Set video=0 to return images from all videos (note these will not all be the same size). Otherwise choose which video to return images from.
    
    Landcover: Videos 1,4
    WV multi-sprectral: Video 5
    WV panchromatic: Videos 1,2,5
    S2: Videos 3,4,5
    """
    def __init__(self, sensor='S2', sites='all', panchromatic=True, video=1, change_labels=[2,3,4,7,8,9,11]):
        
        self.sensor = sensor
        
        self.accepted_labels = change_labels ### only take contruction based labels, ignore "transient construction"
        
        if sites == 'all':
            sites = ['AE-Dubai-0001',
                     'BR-Rio-0270',
                     'BR-Rio-0277',
                     'KR-Pyeongchang-S2',
                     'KR-Pyeongchang-WV',
                     'US-Waynesboro-0001']
        
        self.video_id = video
        self.root = '/u/eag-d1/data/watch/smart_watch_dvc/drop0_aligned/'        
        self.json_file = osp.join(self.root, 'data.kwcoco.json')        
        dset = kwcoco.CocoDataset(self.json_file)
        
        if self.video_id:
            video_list = dset.images().lookup('video_id', keepid=True)
            video_ids = [ID for ID in video_list if video_list[ID] == int(self.video_id)]
        else:
            video_ids = [ID for ID in dset.images().lookup('video_id', keepid=True)]
        
        sensor_list = dset.images().lookup('sensor_coarse', keepid=True)
        sensor_ids = [ID for ID in sensor_list if sensor_list[ID] == sensor]

        
#         region_list = dset.images().lookup('site_tag', keepid=True)
#         region_ids = [ID for ID in region_list if region_list[ID] in sites]

        if 'WV' == sensor:
            band_list = dset.images().lookup('num_bands', keepid=True)
            pan_ids = [ID for ID in band_list if band_list[ID] == 1]
            ms_ids = [ID for ID in band_list if band_list[ID] == 8]
            
            if panchromatic:
                sensor_ids = [ids for ids in sensor_ids if ids in pan_ids]
                self.ms = False
            else:
                sensor_ids = [ids for ids in sensor_ids if ids in ms_ids]
                self.ms = True
   
        self.dset_ids = sorted([ids for ids in sensor_ids if ids in video_ids])

        self.annotations = dset.annots
        self.images = dset.images(self.dset_ids)        
    
        self.dset = dset
    
    def __len__(self):
        return len(self.dset_ids)

    def __getitem__(self, idx):
        annot_ids = 1 + np.where(np.array(self.annotations().get('image_id')) == self.images.get('id')[idx])[0]

        annotations = self.annotations(annot_ids)
        
        bbox = annotations.lookup('bbox')
        segmentation = [x['exterior'] for x in annotations.lookup('segmentation')]
        category_id = annotations.lookup('category_id')
        
        filename = osp.join(self.root, self.images.lookup('file_name')[idx])
        acquisition_date = self.images.lookup('date_captured')[idx]
#         region = self.images.lookup('site_tag')[idx]
        
        im = tifffile.imread(filename)
        im = torch.tensor(im.astype('int16'))
        
        if len(im.shape) < 3:
            im = im.unsqueeze(0)
            if self.sensor == 'WV':
                im = im / 2048. #rough normalization
            else:
                im = im / 32000. #rough normalization
        
        else:
            im = im.permute(2,0,1)
            if self.sensor == 'S2':
                im = im / 255.
            elif self.sensor == 'WV':
                im = im / 2048.
        
        if annotations.get('image_id'):
            if not self.images.get('id')[idx]==annotations.get('image_id')[0]:
                print(annotations.get('image_id'))
                print(self.images.get('id')[idx])
        
        timestamp = self.images.lookup('timestamp')[idx]
        #assert(self.images.get('id')[idx]==annotations.get('image_id')[0])
        
        annotations = {#'region': region, 
                         'bbox': bbox, 
                         'segmentation': segmentation, 
                         'category_id': category_id, 
                         'video_id': self.images.lookup('video_id')[idx],
                         'frame_index': self.images.lookup('frame_index')[idx],
                      'width': self.images.lookup('width')[idx],
                       'height': self.images.lookup('height')[idx],
                      'timestamp': timestamp 
                      }
        
        #####create segmentation mask
        segments = sorted(list(zip(annotations['category_id'], annotations['segmentation'])))
        segments = [segs for segs in segments if segs[0] in self.accepted_labels]

        combined = []
        for segment in segments:

            x, y = np.meshgrid(np.arange(annotations['width']), np.arange(annotations['height'])) # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x,y)).T 

            p = Path(segment[1]) # make a polygon
            grid = p.contains_points(points)
            mask = grid.reshape(annotations['height'], annotations['width']) # now you have a mask with points inside a polygon
            mask = torch.tensor(mask).float()*segment[0]
            combined.append(mask.unsqueeze(0))

        overall_mask = torch.max(torch.cat(combined, dim=0), dim =0)[0]   
        #####
        
        return im, overall_mask, acquisition_date, annotations


class drop0_aligned(torch.utils.data.Dataset):
    """Dataset compatible with drop0_aligned_v2 (now just drop0_aligned on DVC). Sites must be a list from the following: 
                    ['AE-Dubai-0001',
                     'BR-Rio-0270',
                     'BR-Rio-0277',
                     'KR-Pyeongchang-S2',
                     'KR-Pyeongchang-WV',
                     'US-Waynesboro-0001']
    or be set as 'all'.
    
    Sensor must be 'WV' (Worldview), 'LC' (LandCover) or 'S2' (Sentinel 2). If 'WV' is chosen, specify whether you want panchromatic (single channel) images, by setting panchromatic=True. If False, 8 channel multi-spectral images will be returned.
    
    In current drop, all Sentinel 2 images are RGB only. Annotations give bounding box/segmentation outlines of construction sites, but we do not have pixel level annotations for building segmentation or change detection.
    
    There are 5 "videos" in the dataset of aligned images across a single location. Set video=0 to return images from all videos (note these will not all be the same size). Otherwise choose which video to return images from.
    
    Landcover: Videos 3,4
    WV multi-sprectral: Video 5
    WV panchromatic: Videos 1,2,5
    S2: Videos 1,4,5
    """
    def __init__(self, sensor='S2', sites='all', panchromatic=True, video=3):
        
        self.sensor = sensor
        
        if sites == 'all':
            sites = ['AE-Dubai-0001',
                     'BR-Rio-0270',
                     'BR-Rio-0277',
                     'KR-Pyeongchang-S2',
                     'KR-Pyeongchang-WV',
                     'US-Waynesboro-0001']
        
        self.video_id = video
        self.root = '/u/eag-d1/data/watch/smart_watch_dvc/drop0_aligned/'         
        self.json_file = osp.join(self.root, 'data.kwcoco.json')        
        dset = kwcoco.CocoDataset(self.json_file)

        if self.video_id:
            video_list = dset.images().lookup('video_id', keepid=True)
            video_ids = [ID for ID in video_list if video_list[ID] == self.video_id]
        else:
            video_ids = [1,2,3,4,5]
            
        sensor_list = dset.images().lookup('sensor_coarse', keepid=True)
        sensor_ids = [ID for ID in sensor_list if sensor_list[ID] == sensor]

        
#         region_list = dset.images().lookup('site_tag', keepid=True)
#         region_ids = [ID for ID in region_list if region_list[ID] in sites]

        if 'WV' == sensor:
            band_list = dset.images().lookup('num_bands', keepid=True)
            pan_ids = [ID for ID in band_list if band_list[ID] == 1]
            ms_ids = [ID for ID in band_list if band_list[ID] == 8]
            
            if panchromatic:
                sensor_ids = [ids for ids in sensor_ids if ids in pan_ids]
                self.ms = False
            else:
                sensor_ids = [ids for ids in sensor_ids if ids in ms_ids]
                self.ms = True
                
        self.dset_ids = sorted([ids for ids in sensor_ids if ids in video_ids])

        self.annotations = dset.annots
        self.images = dset.images(self.dset_ids)        
    
        self.dset = dset
    
    def __len__(self):
        return len(self.dset_ids)

    def __getitem__(self, idx):
        annot_ids = 1 + np.where(np.array(self.annotations().get('image_id')) == self.images.get('id')[idx])[0]

        annotations = self.annotations(annot_ids)
        
        bbox = annotations.lookup('bbox')
        segmentation = [x['exterior'] for x in annotations.lookup('segmentation')]
        category_id = annotations.lookup('category_id')
        
        filename = osp.join(self.root, self.images.lookup('file_name')[idx])
        acquisition_date = self.images.lookup('date_captured')[idx]
        region = self.images.lookup('site_tag')[idx]
        
        im = tifffile.imread(filename)
        im = torch.tensor(im.astype('int16'))
        
        if len(im.shape) < 3:
            im = im.unsqueeze(0)
            if self.sensor == 'WV':
                im = im / 2000. #rough normalization
            else:
                im = im / 32000. #rough normalization
        
        else:
            im = im.permute(2,0,1)
            if self.sensor == 'S2':
                im = im / 255.
            elif self.sensor == 'WV':
                im = im / 2000.
        
        if annotations.get('image_id'):
            if not self.images.get('id')[idx]==annotations.get('image_id')[0]:
                print(annotations.get('image_id'))
                print(self.images.get('id')[idx])
        
        timestamp = self.images.lookup('timestamp')[idx]
        #assert(self.images.get('id')[idx]==annotations.get('image_id')[0])
        
        annotations = {'region': region, 
                         'bbox': bbox, 
                         'segmentation': segmentation, 
                         'category_id': category_id, 
                         'video_id': self.images.lookup('video_id')[idx],
                         'frame_index': self.images.lookup('frame_index')[idx],                         
                      }
        
        return im, acquisition_date, timestamp, annotations





