"""
SeeAlso:
    * ~/data/dvc-repos/smart_expt_dvc/models/kitware/xview_dino/package_trained_model.py

Notes:
    # To test if mmcv is working on your machine:

    python -c "from mmcv.ops import multi_scale_deform_attn"
"""
#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub
from torch.utils import data


class BuildingDetectorConfig(scfg.DataConfig):
    coco_fpath = scfg.Value(None, help='input kwcoco dataset')
    out_coco_fpath = scfg.Value(None, help='output')
    package_fpath = scfg.Value(None, help='pytorch packaged model')
    data_workers = 2
    window_dims = (1024, 1024)
    fixed_resolution = "1GSD"
    batch_size = 1
    window_overlap = 0.5
    device = scfg.Value(0)
    select_images = None
    track_emissions = True


class WrapperDataset(data.Dataset):

    def __init__(self, subdset):
        self.subdset = subdset

    def __len__(self):
        return len(self.subdset)

    def __getitem__(self, index):
        item = self.subdset[index]
        chw, _ = dino_preproc_item(item)
        item['chw'] = chw
        return item


def main(cmdline=1, **kwargs):
    """
    Ignore:
        /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/kitware/xview_dino_detector/checkpoint_best_regular.pth

    Example:
        >>> # xdoctest: +SKIP
        >>> from geowatch.tasks.dino_detector.predict import *  # NOQA
        >>> import ubelt as ub
        >>> import geowatch
        >>> import kwcoco
        >>> dvc_data_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> dvc_expt_dpath = geowatch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> coco_fpath = dvc_data_dpath / 'Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip'
        >>> package_fpath = dvc_expt_dpath / 'models/kitware/xview_dino.pt'
        >>> out_coco_fpath = ub.Path.appdir('geowatch/tests/dino/doctest0').ensuredir() / 'pred_boxes.kwcoco.zip'
        >>> kwargs = {
        >>>     'coco_fpath': coco_fpath,
        >>>     'package_fpath': package_fpath,
        >>>     'out_coco_fpath': out_coco_fpath,
        >>>     'fixed_resolution': '10GSD',
        >>>     'window_dims': (256, 256),
        >>> }
        >>> cmdline = 0
        >>> _ = main(cmdline=cmdline, **kwargs)
        >>> out_dset = kwcoco.CocoDataset(out_coco_fpath)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> import kwimage
        >>> kwplot.plt.ion()
        >>> gid = out_dset.images()[0]
        >>> annots = out_dset.annots(gid=gid)
        >>> dets = annots.detections
        >>> list(map(len, out_dset.images().annots))
        >>> config = out_dset.dataset['info'][-1]['properties']['config']
        >>> delayed = out_dset.coco_image(gid).imdelay(channels='red|green|blue', resolution=config['fixed_resolution'])
        >>> rgb = kwimage.normalize_intensity(delayed.finalize())
        >>> import kwplot
        >>> kwplot.plt.ion()
        >>> kwplot.imshow(rgb, doclf=1)
        >>> top_dets = dets.compress(dets.scores > 0.2)
        >>> top_dets.draw()
    """
    import rich
    config = BuildingDetectorConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    rich.print('config = ' + ub.urepr(config, nl=1))

    from geowatch.tasks.fusion.utils import load_model_from_package
    import torch
    device = config.device
    model = load_model_from_package(config.package_fpath)
    model = model.eval()
    model = model.to(device)
    model.device = device
    # Specific hacks for this specific model
    model.building_id = ub.invert_dict(model.id2name)['Building']

    from geowatch.tasks.fusion.datamodules import kwcoco_datamodule
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
        include_sensors=['WV', 'WV1'],
        channels="(WV):red|green|blue,(WV,WV1):pan",
    )
    datamodule.setup('test')

    torch_dataset = datamodule.torch_datasets['test']
    dino_dset = WrapperDataset(torch_dataset)
    # loader = torch_dataset.make_loader()

    # from geowatch.tasks.fusion.datamodules.kwcoco_dataset import worker_init_fn
    loader = torch.utils.data.DataLoader(
        dino_dset, batch_size=config.batch_size, num_workers=config.data_workers,
        shuffle=False, pin_memory=False,
        # worker_init_fn=worker_init_fn,
        collate_fn=ub.identity,  # disable collation
    )
    batch_iter = iter(loader)

    coco_dset = torch_dataset.sampler.dset

    from kwutil import util_progress
    from geowatch.utils import process_context
    proc = process_context.ProcessContext(
        name='box.predict',
        config=dict(config),
        track_emissions=config.track_emissions,
    )
    proc.start()
    proc.add_disk_info(coco_dset.fpath)

    # pman = util_progress.ProgressManager('progiter')
    import kwimage
    pman = util_progress.ProgressManager()
    gid_to_dets_accum = ub.ddict(list)

    images = coco_dset.images()
    gid_to_cocoimg = ub.dzip(images, images.coco_images)
    # torch.set_grad_enabled(False)
    with pman, torch.no_grad():
        for batch in pman.progiter(batch_iter, total=len(loader), desc='🦖 dino box detector'):
            batch_dets = dino_predict(model, batch)
            # TODO: downweight the scores of boxes on the edge of the window.
            for item, dets in zip(batch, batch_dets):
                frame0 = item['frames'][0]
                gid = frame0['gid']

                # Compute the transform from this window outspace back to image
                # space.
                sl_y, sl_x = frame0['output_space_slice']
                offset_x = sl_x.start
                offset_y = sl_y.start
                vidspace_offset = (offset_x, offset_y)
                scale_out_from_vid = frame0['scale_outspace_from_vid']
                scale_vid_from_out = 1 / scale_out_from_vid
                warp_vid_from_out = kwimage.Affine.affine(
                    scale=scale_vid_from_out,
                    offset=vidspace_offset)
                coco_img = gid_to_cocoimg[gid]
                warp_img_from_vid = coco_img.warp_img_from_vid
                warp_img_from_out = warp_img_from_vid @ warp_vid_from_out

                imgspace_dets = dets.warp(warp_img_from_out)
                gid_to_dets_accum[gid].append(imgspace_dets)

    unseen_gids = set(coco_dset.images()) - set(gid_to_dets_accum)
    print('unseen_gids = {}'.format(ub.urepr(unseen_gids, nl=1)))

    coco_dset.clear_annotations()

    obj = proc.stop()
    from kwcoco.util import util_json
    obj = util_json.ensure_json_serializable(obj)

    out_dset = coco_dset.copy()

    # TODO: graceful bundle changes
    out_dset.reroot(absolute=True)
    out_dset.dataset['info'].append(obj)

    out_fpath = ub.Path(config.out_coco_fpath)
    out_fpath.parent.ensuredir()
    out_dset.fpath = out_fpath
    out_dset._ensure_json_serializable()

    import kwimage
    gid_to_dets = ub.udict(gid_to_dets_accum).map_values(kwimage.Detections.concatenate)
    for gid, dets in gid_to_dets.items():
        for cls in dets.classes:
            out_dset.ensure_category(cls)
        dets = dets.non_max_supress(thresh=0.2, perclass=True)
        for ann in dets.to_coco(dset=out_dset):
            out_dset.add_annotation(**ann, image_id=gid)

    # Hack, could make filtering in the dataloader easier.
    images = out_dset.images()
    wv_images = images.compress([
        s == 'WV' for s in images.lookup('sensor_coarse')])
    out_dset = out_dset.subset(wv_images)

    pred_dpath = ub.Path(out_dset.fpath).parent.absolute()
    rich.print(f'Pred Dpath: [link={pred_dpath}]{pred_dpath}[/link]')
    out_dset.fpath = out_fpath
    out_dset.dump(out_dset.fpath, indent='    ')
    return out_dset


def dino_preproc_item(item):
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
    hardcoded_mean = [0.485, 0.456, 0.406]
    hardcoded_std = [0.229, 0.224, 0.225]
    chw = F.to_tensor(normed_rgb)
    chw = F.normalize(chw, mean=hardcoded_mean, std=hardcoded_std)
    chw = torch.nan_to_num(chw)
    # normed_rgb = np.nan_to_num(normed_rgb)
    return chw, normed_rgb


def dino_predict(model, batch):
    import torch
    import kwimage

    dino_batch_items = []
    for item in batch:
        chw = item['chw']
        dino_batch_items.append(chw)

    dino_batch = torch.stack(dino_batch_items, dim=0)
    device = model.device

    bchw = dino_batch.to(device)

    raw_output = model.forward(bchw)
    raw_output['pred_boxes'].shape
    target_sizes = torch.Tensor([chw.shape[1:3]]).to(device)

    outputs = model.postprocessors['bbox'](raw_output, target_sizes)

    batch_dets = []
    for output in outputs:
        output = outputs[0]
        dets = kwimage.Detections(
            boxes=kwimage.Boxes(output['boxes'].cpu().numpy(), 'ltrb'),
            class_idxs=output['labels'].cpu().numpy(),
            scores=output['scores'].cpu().numpy(),
            classes=list(model.id2name.values()),
        )
        # print(dets)
        FILTER_NON_BUILDING = 0
        if FILTER_NON_BUILDING:
            dets = dets.compress(dets.class_idxs == model.building_id)
        # Do a very small threshold first
        dets = dets.compress(dets.scores > 0.01)

    batch_dets.append(dets)
    return batch_dets


if __name__ == '__main__':
    """

    CommandLine:
        xdoctest -m geowatch.tasks.dino_detector.predict
    """
    main()
