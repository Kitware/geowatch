#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub
from torch.utils import data


if 1:
    try:
        PARENT_DPATH = ub.Path(__file__).parent
    except NameError:
        from watch.tasks.sam import predict as this_mod
        PARENT_DPATH = ub.Path(this_mod.__file__)
    TPL_DPATH = PARENT_DPATH / 'tpl'

    import sys
    import os
    sys.path.append(os.fspath(TPL_DPATH / 'segment_anything'))
    import segment_anything


class SAMConfig(scfg.DataConfig):
    weights_fpath = None
    coco_fpath = scfg.Value(None, help='input kwcoco dataset')
    out_coco_fpath = scfg.Value(None, help='output')
    # package_fpath = scfg.Value(None, help='pytorch packaged model')

    data_workers = 2
    window_dims = (1024, 1024)
    fixed_resolution = "10GSD"
    batch_size = 1
    window_overlap = 0.33333
    device = scfg.Value(0)
    select_images = None
    track_emissions = True
    io_workers = 0
    assets_dname = 'sam'


def rgb_from_kwcoco_item(item):
    import torchvision.transforms.functional as F
    import kwimage
    import torch
    # import numpy as np
    frames = item['frames']
    modes = frames[0]['modes']

    if 'red|green|blue' in modes:
        chw = modes['red|green|blue']
        is_nan = torch.isnan(chw)
        rgb_nan_frac = is_nan.sum() / is_nan.numel()
    else:
        rgb_nan_frac = 1.0

    # print('----')
    # print(f'rgb_nan_frac={rgb_nan_frac}')
    if rgb_nan_frac >= 0.0:
        # fallback on pan
        if 'pan' in modes:
            pan_chw = modes['pan']
            pan_is_nan = torch.isnan(pan_chw)
            pan_nan_frac = pan_is_nan.sum() / pan_is_nan.numel()
            # print(f'pan_nan_frac={pan_nan_frac}')
            if rgb_nan_frac > pan_nan_frac:
                pan_hwc = pan_chw.permute(1, 2, 0).cpu().numpy()
                pan_hwc = kwimage.atleast_3channels(pan_hwc)
                chw = torch.Tensor(pan_hwc).permute(2, 0, 1)

    hwc = chw.permute(1, 2, 0).cpu().numpy()
    normed_rgb = kwimage.normalize_intensity(hwc)
    chw = F.to_tensor(normed_rgb)
    return chw


class WrapperDataset(data.Dataset):

    def __init__(self, subdset):
        self.subdset = subdset

    def __len__(self):
        return len(self.subdset)

    def __getitem__(self, index):
        import kwimage
        import torch
        import numpy as np
        item = self.subdset[index]
        chw = rgb_from_kwcoco_item(item)
        hwc = chw.permute(1, 2, 0).cpu().numpy()
        norm01_resized, info = kwimage.imresize(
            hwc, dsize=(1024, 1024), interpolation='lanczos',
            letterbox=True, border_value=np.nan, return_info=True)
        norm01_resized = norm01_resized.clip(0, 1)
        # isnan_resized = np.isnan(norm01_resized)
        norm255_resized = (norm01_resized * 255).round().astype(np.uint8)
        input_image_torch = torch.as_tensor(norm255_resized)
        # device=sam_model.device)
        chw255 = input_image_torch.permute(2, 0, 1).contiguous()
        item['chw255'] = chw255
        item['resize_info'] = info
        # input_image = sam_model.preprocess(input_image_torch)
        # input_image[:, :, 0:h, 0:w][nan_mask_bchw] = 0
        # input_image.nan_to_num_()
        # print(f'input_image_torch.shape={input_image_torch.shape}')
        # print(f'input_image.shape={input_image.shape}')
        return item


class DenseFeaturePredictor:
    """
    Base class for computing per-image dense features
    """
    short_code = NotImplemented
    chan_code = NotImplemented
    proc_name = NotImplemented

    def __init__(self, config):
        self.config = config

    def setup(self):
        from watch.tasks.fusion.datamodules import kwcoco_datamodule
        from watch.tasks.fusion.coco_stitcher import CocoStitchingManager
        from watch.utils import util_parallel
        from watch.utils import process_context

        config = self.config
        datamodule = kwcoco_datamodule.KWCocoVideoDataModule(
            test_dataset=config.coco_fpath,
            batch_size=config.batch_size,
            fixed_resolution=config.fixed_resolution,
            num_workers=config.data_workers,
            window_dims=config.window_dims,
            select_images=config.select_images,
            window_overlap=config.window_overlap,
            force_bad_frames=True,
            resample_invalid_frames=0,
            time_steps=1,
            channels="red|green|blue,pan",
        )
        datamodule.setup('test')
        self.datamodule = datamodule

        torch_dataset = self.datamodule.torch_datasets['test']
        coco_dset = torch_dataset.sampler.dset
        coco_dset.clear_annotations()
        self.out_dset = coco_dset.copy()
        self.out_dset.out_fpath = ub.Path(config.out_coco_fpath)
        # TODO: graceful bundle changes
        self.out_dset.reroot(absolute=True)

        self.proc_context = process_context.ProcessContext(
            name='box.predict',
            config=dict(config),
            track_emissions=config.track_emissions,
        )

        self.wrapper_dset = WrapperDataset(torch_dataset)

        writer_queue = util_parallel.BlockingJobQueue(max_workers=config.io_workers)
        self.coco_stitcher = CocoStitchingManager(
            self.out_dset,
            short_code=self.short_code,
            chan_code=self.chan_code,
            stiching_space='image',
            writer_queue=writer_queue,
            expected_minmax=(0, 1),
            assets_dname=config.assets_dname or self.short_code,
        )

        device = 0
        sam_model = segment_anything.sam_model_registry['vit_h'](checkpoint=config.weights_fpath)
        sam_model = sam_model.eval()
        sam_model = sam_model.to(device)
        self.sam_model = sam_model

    def run(self):
        import rich
        from torch.utils import data as torch_data
        from watch.utils import util_progress
        from kwcoco.util import util_json
        import torch

        self.proc_context.add_disk_info(self.out_dset.fpath)
        self.proc_context.start()

        loader = torch_data.DataLoader(
            self.wrapper_dset, batch_size=self.config.batch_size,
            num_workers=self.config.data_workers,
            shuffle=False, pin_memory=False,
            # worker_init_fn=worker_init_fn,
            collate_fn=ub.identity,  # disable collation
        )
        batch_iter = iter(loader)
        # torch.set_grad_enabled(False)

        pman = util_progress.ProgressManager()
        with pman, torch.no_grad():
            prog = pman.progiter(batch_iter, total=len(loader),
                                  desc=f'Predict {self.short_code}')
            for batch in prog:
                self.predict_batch(batch)

        for gid in self.coco_stitcher.managed_image_ids():
            self.coco_stitcher.submit_finalize_image(gid)
        self.coco_stitcher.writer_queue.wait_until_finished()

        obj = self.proc_context.stop()
        obj = util_json.ensure_json_serializable(obj)
        self.out_dset.dataset['info'].append(obj)

        ub.Path(self.out_dset.fpath).parent.ensuredir()
        self.out_dset._ensure_json_serializable()

        pred_dpath = ub.Path(self.out_dset.fpath).parent
        print('self.out_dset.fpath = {}'.format(ub.urepr(self.out_dset.fpath, nl=1)))
        rich.print(f'Pred Dpath: [link={pred_dpath}]{pred_dpath}[/link]')
        self.out_dset.dump(indent='    ')
        print('Finished predict')

    def predict_batch(self, batch):
        raise NotImplementedError


class SAMFeaturePredictor(DenseFeaturePredictor):
    short_code = 'sam'
    proc_name = 'watch.tasks.sam.predict'
    chan_code = 'sam.0:256'

    def __init__(self, config):
        super().__init__(config)

    def predict_batch(self, batch):
        import torch
        import kwimage
        batch_chw255 = []
        for item in batch:
            chw255 = item['chw255']
            batch_chw255.append(chw255)
        bchw255 = torch.stack(batch_chw255, dim=0)

        bchw255 = bchw255.to(self.sam_model.device)
        input_image = self.sam_model.preprocess(bchw255)
        input_image.nan_to_num_()
        embedding_bchw = self.sam_model.image_encoder(input_image)

        feat_bchw = embedding_bchw.detach().cpu().numpy()

        for item, feat_chw in zip(batch, feat_bchw):
            frame0 = item['frames'][0]
            frame0['output_dims']
            fake_output_dsize = kwimage.Box.from_dsize(frame0['output_image_dsize'])
            fake_outspace_slice = kwimage.Box.from_slice(frame0['output_space_slice'])
            # frame0['saliency_output_dims']

            gid = frame0['gid']
            feat_hwc = feat_chw.transpose(1, 2, 0)

            coco_img = self.out_dset.coco_image(gid)
            fakeoutspace_from_outspace = kwimage.Affine.scale(16)
            outspace_from_fakeoutspace = fakeoutspace_from_outspace.inv()

            output_dsize = fake_output_dsize.warp(outspace_from_fakeoutspace).quantize()
            asset_dsize = (output_dsize.width, output_dsize.height)

            fakeoutspace_from_vid = kwimage.Affine.coerce(scale=frame0['scale_outspace_from_vid'])
            vid_from_fakeoutspace = fakeoutspace_from_vid.inv()
            img_from_fakeoutspace = coco_img.warp_img_from_vid @ vid_from_fakeoutspace
            img_from_outspace = img_from_fakeoutspace @ fakeoutspace_from_outspace
            outspace_from_img = img_from_outspace.inv()

            resize_offset = item['resize_info']['offset']
            resize_scale = item['resize_info']['scale']
            undo_resize = kwimage.Affine.coerce(scale=resize_scale,
                                                offset=resize_offset).inv()
            if not undo_resize.isclose_identity():
                kwimage.warp_affine
                raise NotImplementedError('todo')

            data = feat_hwc
            img_slice = fake_outspace_slice.warp(img_from_fakeoutspace).quantize().clip(0, 0, None, None).to_slice()
            self.coco_stitcher.accumulate_image(
                gid, img_slice, data, asset_dsize=asset_dsize,
                scale_asset_from_stitchspace=outspace_from_img.decompose()['scale'])


def main(cmdline=1, **kwargs):
    """

    CommandLine:
        DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
        DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

    CommandLine:
        TEST_SAM=1 xdoctest -m watch.tasks.sam.predict main

    Example:
        >>> # xdoctest: +REQUIRES(env:TEST_SAM)
        >>> from watch.tasks.sam.predict import *  # NOQA
        >>> import watch
        >>> dvc_expt_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> weights_fpath = dvc_expt_dpath / 'models/sam/sam_vit_h_4b8939.pth'
        >>> coco_fpath = dvc_data_dpath / 'Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip'
        >>> kwargs = dict(weights_fpath=weights_fpath, coco_fpath=coco_fpath, out_coco_fpath=(dvc_data_dpath / 'Drop6-MeanYear10GSD-V2/_test.kwcoco.zip'))
        >>> cmdline = 0
        >>> main(cmdline, **kwargs)

    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict()
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = SAMConfig.cli(cmdline=cmdline, data=kwargs, strict=True)

    import rich
    rich.print('config = ' + ub.urepr(config, nl=1))

    predictor = SAMFeaturePredictor(config)
    predictor.setup()
    predictor.run()

    if 0:
        import kwcoco
        coco_dset = kwcoco.CocoDataset.coerce(config.coco_fpath)

        coco_img = coco_dset.images().coco_images[0]
        delayed = coco_img.imdelay(channels='red|green|blue')
        data = delayed.finalize()

        import kwimage
        import numpy as np
        norm01 = kwimage.normalize_intensity(data)
        norm255 = kwimage.ensure_uint255(np.nan_to_num(norm01))

        sam_model = segment_anything.sam_model_registry['vit_h'](checkpoint=config.weights_fpath)
        sam_model = sam_model.eval()
        device = 0
        sam_model = sam_model.to(device)

        import torch
        torch.set_grad_enabled(False)

        if 0:
            sam_predictor = segment_anything.SamPredictor(sam_model)
            sam_predictor.set_image(norm255)
            embedding_bchw_v1 = sam_predictor.get_image_embedding()
            embedding_hwc_v1 = embedding_bchw_v1[0].permute(1, 2, 0).detach().cpu().numpy()
            embedding_viz_v1 = kwimage.normalize_intensity(embedding_hwc_v1[..., 0:3])

        if 1:
            import torch
            norm01_resized, info = kwimage.imresize(
                norm01, dsize=(1024, 1024), interpolation='lanczos',
                letterbox=True, border_value=np.nan, return_info=True)
            norm01_resized = norm01_resized.clip(0, 1)
            # isnan_resized = np.isnan(norm01_resized)
            norm255_resized = (norm01_resized * 255)
            input_image_torch = torch.as_tensor(norm255_resized, device=sam_model.device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
            input_image = sam_model.preprocess(input_image_torch)
            # input_image[:, :, 0:h, 0:w][nan_mask_bchw] = 0
            input_image.nan_to_num_()
            print(f'input_image_torch.shape={input_image_torch.shape}')
            print(f'input_image.shape={input_image.shape}')
            embedding_bchw = sam_model.image_encoder(input_image)
            embedding_hwc = embedding_bchw[0].permute(1, 2, 0).detach().cpu().numpy()
            embedding_viz = kwimage.normalize_intensity(embedding_hwc[..., 0:3])

        if 0:
            import kwplot
            kwplot.autompl()
            kwplot.figure(fnum=1, doclf=True)
            kwplot.imshow(norm255, pnum=(1, 3, 1))
            kwplot.imshow(embedding_viz, pnum=(1, 3, 2))
            kwplot.imshow(embedding_viz_v1, pnum=(1, 3, 3))


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/watch/tasks/sam/predict.py
        python -m predict
    """
    main()
