import kwcoco
import kwimage
import kwarray
import ndsampler
import pathlib
import numpy as np
from torch.utils import data
import torch
import einops

class OneraDataset(data.Dataset):
    # TODO: add torchvision.transforms or albumentations
    def __init__(self, sampler, sample_shape, channels=None):
        self.sampler = sampler
        self.sample_shape = sample_shape
        self.channels = channels
        self.sample_grid = self.sampler.new_sample_grid("video_detection", self.sample_shape)["positives"]
    
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
        for raw_frame, raw_dets in zip(raw_frame_list, raw_det_list):
            frame = raw_frame.astype(np.float32)
            dets = raw_dets
            input_dsize = self.sample_shape[-2:]

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
        
        # compute change from masks
        changes = frame_masks[1:] != frame_masks[:-1]

        example = {
            "images": frame_ims,
            "changes": changes,
        }
        
        return example