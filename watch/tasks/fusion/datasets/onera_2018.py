import kwcoco
import kwimage
import kwarray
import ndsampler
import pathlib
import numpy as np
from torch.utils import data
import torch
import einops
import itertools as it

class OneraDataset(data.Dataset):
    # TODO: add torchvision.transforms or albumentations
    def __init__(self, sampler, sample_shape, channels=None, mode="fit"):
        self.sampler = sampler
        self.sample_shape = sample_shape
        self.channels = channels
        self.mode = mode
        
        full_sample_grid = self.sampler.new_sample_grid("video_detection", self.sample_shape)
        self.sample_grid = list(it.chain(
            full_sample_grid["positives"], 
            full_sample_grid["negatives"],
        ))
        
        example_to_query = self.__getitem__(0)
        if self.mode == "predict":
            self.num_channels = example_to_query.shape[1]
        else:
            self.num_channels = example_to_query["images"].shape[1]
        
    def compute_stats(self, num_examples=1):
        
        # get some samples, reshape/flatten so that channels lead
        images = torch.stack([
            self.__getitem__(idx)["images"]
            for idx in range(num_examples)
        ], dim=0).numpy()
        channel_images = einops.rearrange(
            images,
            "b t c h w -> c (b t h w)",
        )
        
        # dump outliers
        cmin, cmax = np.percentile(channel_images, [2, 98], axis=1)
        channel_images[channel_images < cmin[:,None]] = np.nan
        channel_images[channel_images > cmax[:,None]] = np.nan
        
        # compute stats
        return channel_images.mean(axis=1), channel_images.std(axis=1)
    
    def __len__(self):
        return len(self.sample_grid)
    
    def __getitem__(self, idx):
        
        # get positive sample definition
        tr = self.sample_grid[idx]
        if self.channels:
            tr["channels"] = self.channels
        
        # collect sample
        sample = self.sampler.load_sample(tr)

        # Access the sampled image and annotation data
        raw_frame_list = sample['im']
        raw_det_list = sample['annots']['frame_dets']

        # Break data down on a per-frame basis so we can apply image-based
        # augmentations.
        frame_ims = []
        frame_masks = []
        for frame, dets in zip(raw_frame_list, raw_det_list):
            frame = frame.astype(np.float32)
            input_dsize = self.sample_shape[-2:]
            
            input_dsize = (
                real if (nominal is None) else nominal
                for nominal, real in zip(input_dsize, frame.shape)
            )

            # Resize the sampled window to the target space for the network
            frame, info = kwimage.imresize(frame, dsize=input_dsize,
                                           interpolation='linear',
                                           return_info=True)
            # Remember to apply any transform to the dets as well
            dets = dets.scale(info['scale'])
            dets = dets.translate(info['offset'])

            frame_mask = np.full(frame.shape[0:2], dtype=np.int32, fill_value=-1)
            ann_polys = dets.data['segmentations'].to_polygon_list()
            ann_aids = dets.data['aids']
            ann_cids = dets.data['cids']

            for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):
                cidx = self.sampler.classes.id_to_idx[cid]
                poly.fill(frame_mask, value=cidx)

            # ensure channel dim is not squeezed
            frame = kwarray.atleast_nd(frame, 3)

            frame_masks.append(frame_mask)
            frame_ims.append(frame)
            
        # stack along temporal axis
        frame_masks = np.stack(frame_masks, axis=0)
        frame_ims = np.stack(frame_ims, axis=0)
        
        # rearrange image axes for pytorch
        frame_ims = einops.rearrange(frame_ims, "t h w c -> t c h w")
        frame_ims = frame_ims / 2000.
        
        # catch nans
        frame_ims[np.isnan(frame_ims)] = -1.
        
        if self.mode == "predict":
            return torch.from_numpy(frame_ims).detach()
        
        # compute change from masks
        changes = frame_masks[1:] != frame_masks[:-1]

        example = {
            "images": torch.from_numpy(frame_ims).detach(),
            "changes": torch.from_numpy(changes).detach().int(),
        }
        
        return example


class SimpleDataset(data.Dataset):
    # TODO: add torchvision.transforms or albumentations
    def __init__(self, sampler, sample_shape, channels=None, mode="fit"):
        self.sampler = sampler
        self.sample_shape = sample_shape
        self.channels = channels
        self.mode = mode
        
        full_sample_grid = self.sampler.new_sample_grid("video_detection", self.sample_shape)
        self.sample_grid = list(it.chain(
            full_sample_grid["positives"], 
            full_sample_grid["negatives"],
        ))
        
        example_to_query = self.__getitem__(0)
        if self.mode == "predict":
            self.num_channels = example_to_query.shape[1]
        else:
            self.num_channels = example_to_query["images"].shape[1]
        
    def compute_stats(self, num_examples=1):
        
        # get some samples, reshape/flatten so that channels lead
        images = torch.stack([
            self.__getitem__(idx)["images"]
            for idx in range(num_examples)
        ], dim=0).numpy()
        channel_images = einops.rearrange(
            images,
            "b t c h w -> c (b t h w)",
        )
        
        # dump outliers
        cmin, cmax = np.percentile(channel_images, [2, 98], axis=1)
        channel_images[channel_images < cmin[:,None]] = np.nan
        channel_images[channel_images > cmax[:,None]] = np.nan
        
        # compute stats
        return channel_images.mean(axis=1), channel_images.std(axis=1)
    
    def __len__(self):
        return len(self.sample_grid)
    
    def __getitem__(self, idx):
        
        # get positive sample definition
        tr = self.sample_grid[idx]
        if self.channels:
            tr["channels"] = self.channels
        
        # collect sample
        sample = self.sampler.load_sample(tr)

        # Access the sampled image and annotation data
        raw_frame_list = sample['im']
        raw_det_list = sample['annots']['frame_dets']

        # Break data down on a per-frame basis so we can apply image-based
        # augmentations.
        frame_ims = []
        frame_masks = []
        for frame, dets in zip(raw_frame_list, raw_det_list):
            frame = frame.astype(np.float32)
            input_dsize = self.sample_shape[-2:]
            
            input_dsize = (
                real if (nominal is None) else nominal
                for nominal, real in zip(input_dsize, frame.shape)
            )

            # Resize the sampled window to the target space for the network
            frame, info = kwimage.imresize(frame, dsize=input_dsize,
                                           interpolation='linear',
                                           return_info=True)
            # Remember to apply any transform to the dets as well
            dets = dets.scale(info['scale'])
            dets = dets.translate(info['offset'])

            frame_mask = np.full(frame.shape[0:2], dtype=np.int32, fill_value=-1)
            ann_polys = dets.data['segmentations'].to_polygon_list()
            ann_aids = dets.data['aids']
            ann_cids = dets.data['cids']

            for poly, aid, cid in zip(ann_polys, ann_aids, ann_cids):
                cidx = self.sampler.classes.id_to_idx[cid]
                poly.fill(frame_mask, value=cidx)

            # ensure channel dim is not squeezed
            frame = kwarray.atleast_nd(frame, 3)

            frame_masks.append(frame_mask)
            frame_ims.append(frame)
            
        # stack along temporal axis
        frame_masks = np.stack(frame_masks, axis=0)
        frame_ims = np.stack(frame_ims, axis=0)
        
        # rearrange image axes for pytorch
        frame_ims = einops.rearrange(frame_ims, "t h w c -> t c h w")
        
        # catch nans
        frame_ims[np.isnan(frame_ims)] = -1.
        
        if self.mode == "predict":
            return torch.from_numpy(frame_ims).detach()

        example = {
            "images": torch.from_numpy(frame_ims).detach(),
            "labels": torch.from_numpy(frame_masks).detach().int()+1,
        }
        
        return example
