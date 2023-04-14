"""
SeeAlso:
    * /data/joncrall/dvc-repos/smart_expt_dvc/models/kitware/xview_dino/package_trained_model.py
"""
#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class BuildingDetectorConfig(scfg.DataConfig):
    coco_fpath = scfg.Value(None, help='input kwcoco dataset')
    out_coco_fpath = scfg.Value(None, help='output')
    package_fpath = scfg.Value(None, help='pytorch packaged model')
    data_workers = 2
    window_dims = (256, 256)
    fixed_resolution = "5GSD"
    batch_size = 1
    device = scfg.Value(0)
    select_images = None
    track_emissions = True


def main(cmdline=1, **kwargs):
    """
    Ignore:
        /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/kitware/xview_building_detector/checkpoint_best_regular.pth

    Example:
        >>> # xdoctest: +SKIP
        >>> from watch.tasks.building_detector.predict import *  # NOQA
        >>> import ubelt as ub
        >>> import watch
        >>> import kwcoco
        >>> dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> dvc_expt_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> coco_fpath = dvc_data_dpath / 'Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip'
        >>> package_fpath = dvc_expt_dpath / 'models/kitware/xview_dino.pt'
        >>> out_coco_fpath = ub.Path.appdir('watch/tests/dino/doctest0').ensuredir() / 'pred_boxes.kwcoco.zip'
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

    from watch.tasks.fusion.utils import load_model_from_package
    import torch
    device = config.device
    model = load_model_from_package(config.package_fpath)
    model = model.eval()
    model = model.to(device)
    model.device = device
    # Specific hacks for this specific model
    model.building_id = ub.invert_dict(model.id2name)['Building']

    from watch.tasks.fusion.datamodules import kwcoco_datamodule
    datamodule = kwcoco_datamodule.KWCocoVideoDataModule(
        test_dataset=config.coco_fpath,
        batch_size=config.batch_size,
        fixed_resolution=config.fixed_resolution,
        num_workers=config.data_workers,
        window_dims=config.window_dims,
        select_images=config.select_images,
        time_steps=1,
        include_sensors=['WV'],
        channels="(WV):red|green|blue",
    )
    datamodule.setup('test')

    loader = datamodule.torch_datasets['test'].make_loader()
    batch_iter = iter(loader)

    coco_dset = loader.dataset.sampler.dset

    from watch.utils import util_progress
    from watch.utils import process_context
    proc = process_context.ProcessContext(
        name='box.predict',
        config=dict(config),
        track_emissions=config.track_emissions,
    )
    proc.start()
    proc.add_disk_info(coco_dset.fpath)

    # pman = util_progress.ProgressManager('progiter')
    pman = util_progress.ProgressManager()
    gid_to_dets_accum = ub.ddict(list)
    # torch.set_grad_enabled(False)
    with pman, torch.no_grad():
        for batch in pman.progiter(batch_iter, total=len(loader), desc='🦖 dino box detector'):
            batch_dets = dino_predict(model, batch)

            # TODO: downweight the scores of boxes on the edge of the window.

            for item, dets in zip(batch, batch_dets):
                frame0 = item['frames'][0]
                gid = frame0['gid']
                sl_y, sl_x = frame0['output_space_slice']
                offset_x = sl_x.start
                offset_y = sl_y.start
                imgspace_dets = dets.translate((offset_x, offset_y))
                gid_to_dets_accum[gid].append(imgspace_dets)

    coco_dset.clear_annotations()

    obj = proc.stop()
    from kwcoco.util import util_json
    obj = util_json.ensure_json_serializable(obj)

    out_dset = coco_dset.copy()

    # TODO: graceful bundle changes
    out_dset.reroot(absolute=True)
    out_dset.dataset['info'].append(obj)

    out_fpath = ub.Path(config.out_coco_fpath)
    out_dset.fpath = out_fpath
    out_dset._ensure_json_serializable()

    import kwimage
    gid_to_dets = ub.udict(gid_to_dets_accum).map_values(kwimage.Detections.concatenate)
    for gid, dets in gid_to_dets.items():
        for cls in dets.classes:
            out_dset.ensure_category(cls)
        for ann in dets.to_coco(dset=out_dset):
            out_dset.add_annotation(**ann, image_id=gid)

    # Hack, could make filtering in the dataloader easier.
    images = out_dset.images()
    wv_images = images.compress([
        s == 'WV' for s in images.lookup('sensor_coarse')])
    out_dset = out_dset.subset(wv_images)

    pred_dpath = ub.Path(out_dset.fpath).parent
    rich.print(f'Pred Dpath: [link={pred_dpath}]{pred_dpath}[/link]')
    out_dset.fpath = out_fpath
    out_dset.dump(out_dset.fpath, indent='    ')
    return out_dset


def dino_preproc_item(item):
    import torchvision.transforms.functional as F
    import kwimage
    import numpy as np
    chw = item['frames'][0]['modes']['red|green|blue']
    hwc = chw.permute(1, 2, 0).cpu().numpy()
    normed_rgb = kwimage.normalize_intensity(hwc)
    normed_rgb = np.nan_to_num(normed_rgb)
    hardcoded_mean = [0.485, 0.456, 0.406]
    hardcoded_std = [0.229, 0.224, 0.225]
    chw = F.to_tensor(normed_rgb)
    chw = F.normalize(chw, mean=hardcoded_mean, std=hardcoded_std)
    return chw, normed_rgb


def dino_predict(model, batch):
    import torch
    import kwimage

    dino_batch_items = []
    for item in batch:
        chw, normed_rgb = dino_preproc_item(item)
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

    batch_dets.append(dets)
    return batch_dets


if __name__ == '__main__':
    """

    CommandLine:
        xdoctest -m watch.tasks.building_detector.predict
    """
    main()
