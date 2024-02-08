#!/usr/bin/env python3
"""
CommandLine:
    # To Execute
    export DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    sdvc request -r aws $DVC_EXPT_DPATH/models/sam/sam_vit_h_4b8939.pth
    xdoctest geowatch.tasks.sam.predict __doc__:0

Example:
    >>> # xdoctest: +REQUIRES(env:DVC_EXPT_DPATH)
    >>> import geowatch
    >>> import kwcoco
    >>> import ubelt as ub
    >>> # Define where to write the output
    >>> output_dpath = ub.Path.appdir('geowatch/tests/sam/demo').ensuredir()
    >>> output_kwcoco_fpath = output_dpath / 'demo_sam.kwcoco.zip'
    >>> # Define the input
    >>> dset = kwcoco.CocoDataset.demo('vidshapes', num_frames=4, num_videos=1)
    >>> dset.reroot(absolute=True)
    >>> dset.fpath = output_dpath / 'sam_input.kwcoco.zip'
    >>> dset.dump()
    >>> input_kwcoco_fpath = dset.fpath
    >>> # The main external data this test needs is the SAM weights
    >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_expt')
    >>> weights_fpath = dvc_dpath / "models/sam/sam_vit_h_4b8939.pth"
    >>> # Setup the arguments to SAM predict
    >>> from geowatch.tasks.sam import predict as sam_predict
    >>> kwargs = {
    >>>     'input_kwcoco': input_kwcoco_fpath,
    >>>     'output_kwcoco': output_kwcoco_fpath,
    >>>     'fixed_resolution': None,
    >>>     'weights_fpath': weights_fpath,
    >>>     'channels': 'r|g|b',
    >>> }
    >>> cmdline = 0
    >>> sam_predict.main(cmdline=cmdline, **kwargs)
    >>> if 1:
    >>>     ub.cmd(f'geowatch visualize {output_kwcoco_fpath} --stack=only --channels "r|g|b,sam.0:3,sam.3:6,sam.6:9"', verbose=3)
"""
import scriptconfig as scfg
import ubelt as ub
from torch.utils import data


class SAMConfig(scfg.DataConfig):
    r"""
    Compute SAM encoder features

    Usage:
        DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
        python -m geowatch.tasks.sam.predict \
            --input_kwcoco <input-kwcoco>
            --output_kwcoco <output-kwcoco>
            --fixed_resolution=None
            --channels="red|green|blue"
            --weights_fpath "$DVC_EXPT_DPATH/models/sam/sam_vit_h_4b8939.pth" \
            --window_overlap=0.33333 \
            --data_workers="2" \
            --io_workers 0 \
            --assets_dname="sam_feats"
    """
    input_kwcoco = scfg.Value(None, help='input kwcoco dataset')
    output_kwcoco = scfg.Value(None, help='output')

    weights_fpath = scfg.Value(None, help='path to pytorch model weights')

    channels = scfg.Value("red|green|blue,pan", help=ub.paragraph(
        '''
        The channels in the dataset to be passed to SAMs rgb inputs
        By default this is red|green|blue or pan, but if you have a different
        channel code for rgb (e.g. r|g|b) then that must be specified.
        In the future we may change this option such that it does a better job
        at inference.
        '''
    ))

    window_dims = (1024, 1024)
    fixed_resolution = "10GSD"
    batch_size = 1
    window_overlap = 0.33333
    device = scfg.Value(0, help='The torch device to predict on')
    select_images = None
    track_emissions = True
    data_workers = 2
    io_workers = 0
    assets_dname = 'sam'


def rgb_from_kwcoco_frame(frame, channel_priority):
    import torchvision.transforms.functional as F
    import kwimage
    import torch
    # import numpy as np
    modes = frame['modes']
    chw = None

    # A default channel_priority would look like:
    # channel_priority = [
    #     {'mode': 'red|green|blue', 'numel': 3},
    #     {'mode': 'pan', 'numel': 1},
    # ]

    best = {
        'chw': None,
        'nan_frac': 1.1,
    }

    for candidate in channel_priority:
        # Find a candidate chanel that the frame has and is not mostly nan
        cand_mode = candidate['mode']
        cand_numel = candidate['numel']
        if cand_mode in modes:
            cand_chw = modes[cand_mode]
            cand_is_nan = torch.isnan(cand_chw)
            cand_nan_frac = cand_is_nan.sum() / cand_is_nan.numel()

            if cand_numel == 1:
                # Convert to 3 channels
                cand_hwc = cand_chw.permute(1, 2, 0).cpu().numpy()
                cand_hwc = kwimage.atleast_3channels(cand_hwc)
                cand_chw = torch.Tensor(cand_hwc).permute(2, 0, 1)

            if cand_nan_frac < best['nan_frac']:
                best = {
                    'chw': cand_chw,
                    'nan_frac': cand_nan_frac,
                }

            if cand_nan_frac == 0:
                # No nans, looks good, stop early
                break

    # if 'red|green|blue' in modes:
    #     chw = modes['red|green|blue']
    #     is_nan = torch.isnan(chw)
    #     rgb_nan_frac = is_nan.sum() / is_nan.numel()
    # else:
    #     rgb_nan_frac = 1.0
    # # print('----')
    # # print(f'rgb_nan_frac={rgb_nan_frac}')
    # if rgb_nan_frac >= 0.0:
    #     # fallback on pan
    #     if 'pan' in modes:
    #         pan_chw = modes['pan']
    #         pan_is_nan = torch.isnan(pan_chw)
    #         pan_nan_frac = pan_is_nan.sum() / pan_is_nan.numel()
    #         # print(f'pan_nan_frac={pan_nan_frac}')
    #         if chw is None or rgb_nan_frac > pan_nan_frac:
    #             pan_hwc = pan_chw.permute(1, 2, 0).cpu().numpy()
    #             pan_hwc = kwimage.atleast_3channels(pan_hwc)
    #             chw = torch.Tensor(pan_hwc).permute(2, 0, 1)

    chw = best['chw']
    if chw is None:
        raise AssertionError('Unable to find an RGB frame')

    hwc = chw.permute(1, 2, 0).cpu().numpy()
    normed_rgb = kwimage.normalize_intensity(hwc)
    chw = F.to_tensor(normed_rgb)
    return chw


class SAMWrapperDataset(data.Dataset):

    def __init__(self, subdset, channel_priority):
        self.subdset = subdset
        self.channel_priority = channel_priority

    def __len__(self):
        return len(self.subdset)

    def __getitem__(self, index):
        import kwimage
        import torch
        import numpy as np
        import einops

        torch_dataset = self.subdset

        item = torch_dataset[index]

        frame0 = item['frames'][0]
        channel_priority = self.channel_priority
        chw = rgb_from_kwcoco_frame(frame0, channel_priority)
        hwc = chw.permute(1, 2, 0).cpu().numpy()
        norm01_resized, resize_info = kwimage.imresize(
            hwc, dsize=(1024, 1024), interpolation='lanczos',
            letterbox=True, border_value=np.nan, return_info=True)
        norm01_resized = norm01_resized.clip(0, 1)
        hwc_isnodata_mask = np.isnan(norm01_resized)

        # Get a nodata mask for the output we expect, Only mask an output block
        # if more than half of its input pixels were nodata.
        factor = 16
        isnodat_mask = hwc_isnodata_mask.all(axis=2).astype(np.uint8)
        isnodata_blocks = einops.rearrange(isnodat_mask, '(h2 s1) (w2 s2) -> h2 w2 (s1 s2)', s1=factor, s2=factor)
        output_isnodata_mask = isnodata_blocks.sum(axis=2) > (factor * factor) // 2

        norm255_resized = (np.nan_to_num(norm01_resized) * 255).round().astype(np.uint8)
        input_image_torch = torch.as_tensor(norm255_resized)

        chw255 = input_image_torch.permute(2, 0, 1).contiguous()

        # coco_dset = torch_dataset.sampler.dset
        gid = frame0['gid']
        # coco_img = coco_dset.coco_image(gid)

        # SAM will always downsample the output by a factor of 16 after it has
        # been resized to 1024x1024. Munge the transforms to account for this.
        fakeoutspace_from_outspace = kwimage.Affine.scale(16)
        outspace_from_fakeoutspace = fakeoutspace_from_outspace.inv()

        # The dataloader is assuming the output will be the same shape as the
        # input but that's not true. We will call this assumed output space the
        # fake outspace. The real outspace is actually downsampled by a factor
        # of 16.

        fake_output_dsize = kwimage.Box.from_dsize(frame0['output_image_dsize'])
        fake_outspace_slice = kwimage.Box.from_slice(frame0['output_space_slice'])
        fakeoutspace_from_vid = kwimage.Affine.coerce(scale=frame0['scale_outspace_from_vid'])

        output_dsize = fake_output_dsize.warp(outspace_from_fakeoutspace).quantize()
        outspace_slice_ = fake_outspace_slice.warp(outspace_from_fakeoutspace)
        asset_slice = outspace_slice_.quantize().clip(0, 0, None, None).to_slice()
        asset_dsize = (output_dsize.width, output_dsize.height)

        vid_from_fakeoutspace = fakeoutspace_from_vid.inv()
        vid_from_outspace = vid_from_fakeoutspace @ fakeoutspace_from_outspace
        outspace_from_vid = vid_from_outspace.inv()

        resize_offset = resize_info['offset']
        resize_scale = resize_info['scale']
        undo_resize = kwimage.Affine.coerce(scale=resize_scale,
                                            offset=resize_offset).inv()
        if not undo_resize.isclose_identity():
            kwimage.warp_affine
            raise NotImplementedError('todo')

        item['chw255'] = chw255
        item['chw_isnodata_mask'] = torch.as_tensor(hwc_isnodata_mask.transpose(2, 1, 0)).contiguous()
        item['output_isnodata_mask'] = np.ascontiguousarray(output_isnodata_mask)
        item['gid'] = gid
        item['asset_slice'] = asset_slice
        item['asset_dsize'] = asset_dsize
        item['scale_asset_from_vid'] = outspace_from_vid.decompose()['scale']
        return item


class DenseFeaturePredictor:
    """
    Base class for computing per-image dense features
    """
    short_code = NotImplemented
    chan_code = NotImplemented
    proc_name = NotImplemented
    WrapperDsetClass = NotImplemented

    def __init__(self, config):
        self.config = config

    def setup(self):
        from geowatch.tasks.fusion.datamodules import kwcoco_datamodule
        from geowatch.tasks.fusion.coco_stitcher import CocoStitchingManager
        from kwutil import util_parallel
        from geowatch.utils import process_context
        import kwcoco

        # put the vendored package into our namespace.
        import geowatch_tpl  # NOQA

        segment_anything = geowatch_tpl.import_submodule('segment_anything')

        config = self.config
        datamodule = kwcoco_datamodule.KWCocoVideoDataModule(
            test_dataset=config.input_kwcoco,
            batch_size=config.batch_size,
            fixed_resolution=config.fixed_resolution,
            num_workers=config.data_workers,
            window_dims=config.window_dims,
            select_images=config.select_images,
            window_overlap=config.window_overlap,
            force_bad_frames=True,
            resample_invalid_frames=0,
            time_steps=1,
            channels=config.channels,
            # channels="red|green|blue,pan",
        )

        # Build a list of channel priorities to indicate what
        # SAM should interpret as RGB or coerce to RGB
        channel_priority = []
        for spec in kwcoco.ChannelSpec.coerce(config.channels).streams():
            channel_priority.append({
                'mode': spec.spec,
                'numel': spec.numel(),
            })

        datamodule.setup('test')
        self.datamodule = datamodule

        torch_dataset = self.datamodule.torch_datasets['test']
        coco_dset = torch_dataset.sampler.dset
        coco_dset.clear_annotations()
        self.out_dset = coco_dset.copy()
        self.out_dset.fpath = ub.Path(config.output_kwcoco)
        # TODO: graceful bundle changes
        self.out_dset.reroot(absolute=True)

        self.proc_context = process_context.ProcessContext(
            name=self.proc_name,
            config=dict(config),
            track_emissions=config.track_emissions,
        )

        # Not sure why targets are not in image-first order, but let's ensure
        # they are so we can fix it.
        torch_dataset.new_sample_grid['targets'] = sorted(
            torch_dataset.new_sample_grid['targets'],
            key=lambda x: (x['video_id'], x['main_gid']))

        self.wrapper_dset = self.WrapperDsetClass(torch_dataset, channel_priority)

        writer_queue = util_parallel.BlockingJobQueue(max_workers=config.io_workers)
        self.coco_stitcher = CocoStitchingManager(
            self.out_dset,
            short_code=self.short_code,
            chan_code=self.chan_code,
            # stiching_space='image',
            stiching_space='video',  # stitch in video space
            writer_queue=writer_queue,
            expected_minmax=(0, 1),
            assets_dname=config.assets_dname or self.short_code,
        )

        self.device = self.config.device
        sam_model = segment_anything.sam_model_registry['vit_h'](checkpoint=config.weights_fpath)
        sam_model = sam_model.eval()
        sam_model = sam_model.to(self.device)
        self.sam_model = sam_model

    def run(self):
        """
        Ignore:
            self = predictor
        """
        import rich
        from torch.utils import data as torch_data
        from kwutil import util_progress
        from kwcoco.util import util_json
        import torch

        out_fpath = ub.Path(self.out_dset.fpath)
        pred_dpath = out_fpath.parent
        pred_dpath.ensuredir()

        self.proc_context.add_disk_info(pred_dpath)
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
        previously_managed_gids = set()
        with pman, torch.no_grad():
            prog = pman.progiter(batch_iter, total=len(loader),
                                  desc=f'Predict {self.short_code}')
            for batch in prog:
                self.predict_batch(batch)

                # Optimization:
                # Because we know this process works 1 image at a time and a
                # new image will only be seen after the current image is
                # completed We can flush all previously managed image ids as
                # soon as we see a new one. This prevents all predictions from
                # living in memory.
                currently_managed_gids = set(self.coco_stitcher.managed_image_ids())
                new_gids = currently_managed_gids - previously_managed_gids
                if new_gids:
                    for gid in previously_managed_gids:
                        self.coco_stitcher.submit_finalize_image(gid)
                    self.coco_stitcher.flush_images()
                previously_managed_gids = set(self.coco_stitcher.managed_image_ids())

        for gid in self.coco_stitcher.managed_image_ids():
            self.coco_stitcher.submit_finalize_image(gid)
        self.coco_stitcher.flush_images()

        obj = self.proc_context.stop()
        obj = util_json.ensure_json_serializable(obj)
        self.out_dset.dataset['info'].append(obj)
        self.out_dset._ensure_json_serializable()

        print(f'out_fpath={out_fpath}')
        rich.print(f'Pred Dpath: [link={pred_dpath}]{pred_dpath}[/link]')
        self.out_dset.dump(indent='    ')
        print('Finished predict')

    def predict_batch(self, batch):
        raise NotImplementedError


class SAMFeaturePredictor(DenseFeaturePredictor):
    short_code = 'sam'
    proc_name = 'geowatch.tasks.sam.predict'
    chan_code = 'sam.0:256'

    WrapperDsetClass = SAMWrapperDataset

    def __init__(self, config):
        super().__init__(config)

    def predict_batch(self, batch):
        import torch
        batch_chw255 = []
        batch_chw_isnodata = []
        for item in batch:
            chw255 = item['chw255']
            chw_isnodata = item['chw_isnodata_mask']
            batch_chw255.append(chw255)
            batch_chw_isnodata.append(chw_isnodata)
        bchw255 = torch.stack(batch_chw255, dim=0)
        bchw_isnodata = torch.stack(batch_chw_isnodata, dim=0)

        bchw_isnodata = bchw_isnodata.to(self.device)
        bchw255 = bchw255.to(self.device)
        input_image = self.sam_model.preprocess(bchw255)
        input_image[bchw_isnodata] = 0
        embedding_bchw = self.sam_model.image_encoder(input_image)

        feat_bchw = embedding_bchw.detach().cpu().numpy()

        for item, feat_chw in zip(batch, feat_bchw):
            data = feat_chw.transpose(1, 2, 0)
            gid = item['gid']
            asset_dsize = item['asset_dsize']
            asset_slice = item['asset_slice']

            # If the input was mostly nodata, mask the output
            output_isnodata_mask = item['output_isnodata_mask']
            data[output_isnodata_mask] = float('nan')

            scale_asset_from_vid = item['scale_asset_from_vid']
            self.coco_stitcher.accumulate_image(
                gid, asset_slice, data, asset_dsize=asset_dsize,
                scale_asset_from_stitchspace=scale_asset_from_vid,
                downweight_edges=True)


def main(cmdline=1, **kwargs):
    """

    CommandLine:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

    CommandLine:
        TEST_SAM=1 xdoctest -m geowatch.tasks.sam.predict main

    Example:
        >>> # xdoctest: +REQUIRES(env:TEST_SAM)
        >>> from geowatch.tasks.sam.predict import *  # NOQA
        >>> import geowatch
        >>> dvc_expt_dpath = geowatch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> dvc_data_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> weights_fpath = dvc_expt_dpath / 'models/sam/sam_vit_h_4b8939.pth'
        >>> input_kwcoco = dvc_data_dpath / 'Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip'
        >>> kwargs = dict(
        >>>     weights_fpath=weights_fpath,
        >>>     input_kwcoco=input_kwcoco,
        >>>     output_kwcoco=(dvc_data_dpath / 'Drop6-MeanYear10GSD-V2/_test.kwcoco.zip')
        >>> )
        >>> cmdline = 0
        >>> main(cmdline, **kwargs)

    Ignore:
        from geowatch.tasks.sam.predict import *  # NOQA
        import geowatch
        dvc_expt_dpath = geowatch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        dvc_data_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        weights_fpath = dvc_expt_dpath / 'models/sam/sam_vit_h_4b8939.pth'
        input_kwcoco = dvc_data_dpath / 'Drop6-MeanYear10GSD-V2/imganns-AE_R001.kwcoco.zip'
        kwargs = dict(weights_fpath=weights_fpath, input_kwcoco=input_kwcoco, output_kwcoco=(dvc_data_dpath / 'Drop6-MeanYear10GSD-V2/imganns-AE_R001_sam.kwcoco.zip'))
        cmdline = 0
        config = SAMConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
        import rich
        rich.print('config = ' + ub.urepr(config, nl=1))
        self = predictor = SAMFeaturePredictor(config)
        predictor.setup()
        # main(cmdline, **kwargs)

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

    # if 0:
    #     import kwcoco
    #     coco_dset = kwcoco.CocoDataset.coerce(config.input_kwcoco)

    #     coco_img = coco_dset.images().coco_images[0]
    #     delayed = coco_img.imdelay(channels='red|green|blue')
    #     data = delayed.finalize()

    #     import kwimage
    #     import numpy as np
    #     norm01 = kwimage.normalize_intensity(data)
    #     norm255 = kwimage.ensure_uint255(np.nan_to_num(norm01))

    #     sam_model = segment_anything.sam_model_registry['vit_h'](checkpoint=config.weights_fpath)
    #     sam_model = sam_model.eval()
    #     device = 0
    #     sam_model = sam_model.to(device)

    #     import torch
    #     torch.set_grad_enabled(False)

    #     if 0:
    #         sam_predictor = segment_anything.SamPredictor(sam_model)
    #         sam_predictor.set_image(norm255)
    #         embedding_bchw_v1 = sam_predictor.get_image_embedding()
    #         embedding_hwc_v1 = embedding_bchw_v1[0].permute(1, 2, 0).detach().cpu().numpy()
    #         embedding_viz_v1 = kwimage.normalize_intensity(embedding_hwc_v1[..., 0:3])

    #     if 1:
    #         import torch
    #         norm01_resized, info = kwimage.imresize(
    #             norm01, dsize=(1024, 1024), interpolation='lanczos',
    #             letterbox=True, border_value=np.nan, return_info=True)
    #         norm01_resized = norm01_resized.clip(0, 1)
    #         # isnan_resized = np.isnan(norm01_resized)
    #         norm255_resized = (norm01_resized * 255)
    #         input_image_torch = torch.as_tensor(norm255_resized, device=sam_model.device)
    #         input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    #         input_image = sam_model.preprocess(input_image_torch)
    #         # input_image[:, :, 0:h, 0:w][nan_mask_bchw] = 0
    #         input_image.nan_to_num_()
    #         print(f'input_image_torch.shape={input_image_torch.shape}')
    #         print(f'input_image.shape={input_image.shape}')
    #         embedding_bchw = sam_model.image_encoder(input_image)
    #         embedding_hwc = embedding_bchw[0].permute(1, 2, 0).detach().cpu().numpy()
    #         embedding_viz = kwimage.normalize_intensity(embedding_hwc[..., 0:3])

    #     if 0:
    #         import kwplot
    #         kwplot.autompl()
    #         kwplot.figure(fnum=1, doclf=True)
    #         kwplot.imshow(norm255, pnum=(1, 3, 1))
    #         kwplot.imshow(embedding_viz, pnum=(1, 3, 2))
    #         kwplot.imshow(embedding_viz_v1, pnum=(1, 3, 3))


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/geowatch/tasks/sam/predict.py
        python -m predict

    Ignore:

        python -m geowatch.tasks.sam.predict \
            --input_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD-V2/imganns-AE_R001.kwcoco.zip" \
            --output_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD-V2/imganns-AE_R001_sam.kwcoco.zip" \
            --weights_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/sam/sam_vit_h_4b8939.pth" \
            --window_overlap=0.33333 \
            --data_workers="2" \
            --io_workers 0 \
            --assets_dname="teamfeats"

        geowatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6-MeanYear10GSD-V2/imganns-AE_R001_sam.kwcoco.zip \
            --channels "red|green|blue,pan,sam.0:3,sam.3:6,sam.6:9" --smart

    """
    main()
