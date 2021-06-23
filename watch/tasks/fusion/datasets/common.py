import kwcoco
import kwimage
import kwarray
import ndsampler
import pathlib
import numpy as np
from torch.utils import data
from torch import nn
import torch
import einops
import itertools as it

class AddPositionalEncoding(nn.Module):
    def __init__(self, dest_dim, dims_to_encode):
        super().__init__()
        self.dest_dim = dest_dim
        self.dims_to_encode = dims_to_encode
        assert self.dest_dim not in self.dims_to_encode
        
    def forward(self, x):

        inds = [
            slice(0, size) if (dim in self.dims_to_encode) else slice(0, 1)
            for dim, size in enumerate(x.shape)
        ]
        inds[self.dest_dim] = self.dims_to_encode

        encoding = torch.cat(torch.meshgrid([
            torch.linspace(0, 1, x.shape[dim]) if (dim in self.dims_to_encode) else torch.tensor(-1.)
            for dim in range(len(x.shape))
        ]), dim=self.dest_dim)[inds]

        expanded_shape = list(x.shape)
        expanded_shape[self.dest_dim] = -1
        x = torch.cat([x, encoding.expand(expanded_shape).type_as(x)], dim=self.dest_dim)
        return x

class VideoDataset(data.Dataset):
    # TODO: add torchvision.transforms or albumentations
    def __init__(self, sampler, sample_shape, channels=None, mode="fit", window_overlap=0, transform=None):
        self.sampler = sampler
        self.sample_shape = sample_shape
        self.channels = channels
        self.mode = mode
        self.transform = transform
        self.window_overlap = window_overlap
        
        full_sample_grid = self.sampler.new_sample_grid("video_detection", self.sample_shape, window_overlap=self.window_overlap)
        self.sample_grid = list(it.chain(
            full_sample_grid["positives"], 
            full_sample_grid["negatives"],
        ))
        
        example_to_query = self.__getitem__(0)
        if self.mode == "predict":
            self.num_channels = example_to_query.shape[1]
        else:
            self.num_channels = example_to_query["images"].shape[1]
    
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
            input_dsize = self.sample_shape[-2:][::-1]
            
            input_dsize = [
                real if (nominal is None) else nominal
                for nominal, real in zip(input_dsize, frame.shape[:2][::-1])
            ]

            # Resize the sampled window to the target space for the network
            frame, info = kwimage.imresize(frame, dsize=input_dsize,
                                           interpolation='linear',
                                           antialias=True,
                                           return_info=True)
            # Remember to apply any transform to the dets as well
            dets = dets.scale(info['scale'])
            dets = dets.translate(info['offset'])

            frame_mask = np.full(frame.shape[:2], dtype=np.int32, fill_value=-1)
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
#         frame_ims = frame_ims / 2000.
        
        # catch nans
        frame_ims[np.isnan(frame_ims)] = -1.

        # convert to tensors
        #frame_ims = torch.from_numpy(frame_ims).detach()
        frame_masks = torch.from_numpy(frame_masks).detach().int()
        
        if self.transform:
            frame_ims = self.transform(frame_ims)
        
        if self.mode == "predict":
            return frame_ims

        example = {
            "images": frame_ims,
            "labels": frame_masks,
        }
        
        return example
