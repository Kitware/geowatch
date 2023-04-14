"""
SeeAlso:
    * /data/joncrall/dvc-repos/smart_expt_dvc/models/kitware/xview_dino/package_trained_model.py
"""
#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class BuildingDetectorConfig(scfg.DataConfig):
    coco_fpath = scfg.Value(None, help='input')
    package_fpath = scfg.Value(None, help='input')
    data_workers = 2
    window_dims = (256, 256)
    fixed_resolution = "5GSD"
    batch_size = 1
    device = scfg.Value(0)


def dino_preproc_item(item):
    import torchvision.transforms.functional as F
    import kwimage
    import numpy as np
    chw = item['frames'][0]['modes']['red|green|blue']
    hwc = chw.permute(1, 2, 0).cpu().numpy()
    normed_rgb = kwimage.normalize_intensity(hwc)
    normed_rgb = np.nan_to_num(normed_rgb)
    item['frames'][0]['THUMB'] = normed_rgb
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
        dets = dets.compress(dets.class_idxs == model.building_id)
    batch_dets.append(dets)
    return batch_dets


def main(cmdline=1, **kwargs):
    """
    Ignore:
        /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/kitware/xview_building_detector/checkpoint_best_regular.pth

    Example:
        >>> # xdoctest: +SKIP
        >>> from watch.tasks.building_detector.predict import *  # NOQA
        >>> import ubelt as ub
        >>> import watch
        >>> dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> dvc_expt_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> coco_fpath = dvc_data_dpath / 'Drop6-MeanYear10GSD-V2/imgonly-KR_R001.kwcoco.zip'
        >>> package_fpath = dvc_expt_dpath / 'models/kitware/xview_dino.pt'
        >>> cmdline = 0
        >>> kwargs = {
        >>>     'coco_fpath': coco_fpath,
        >>>     'package_fpath': package_fpath,
        >>> }
        >>> main(cmdline=cmdline, **kwargs)
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
        time_steps=1,
        fixed_resolution=config.fixed_resolution,
        num_workers=config.data_workers,
        window_dims=config.window_dims,
        include_sensors=['WV'],
        channels="(WV):red|green|blue",
    )
    datamodule.setup('test')

    loader = datamodule.torch_datasets['test'].make_loader()
    batch_iter = iter(loader)

    gid_to_dets_accum = ub.ddict(list)

    from watch.utils import util_progress
    pman = util_progress.ProgressManager('progiter')
    # torch.set_grad_enabled(False)
    with pman, torch.no_grad():
        for batch in pman.progiter(batch_iter, total=len(loader), desc='box detector'):
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

    import kwimage
    gid_to_dets = ub.udict(gid_to_dets_accum).map_values(kwimage.Detections.concatenate)

    if 0:
        gid = ub.peek(gid_to_dets)
        gid = list(gid_to_dets)[1]
        dets = gid_to_dets[gid]
        coco_dset = loader.dataset.sampler.dset
        delayed = coco_dset.coco_image(gid).imdelay(channels='red|green|blue', resolution=config.fixed_resolution)
        rgb = kwimage.normalize_intensity(delayed.finalize())

        # dets = batch_dets[0]
        # item = batch[0]
        # rgb = item['frames'][0]['THUMB']

        import kwplot
        kwplot.plt.ion()

        kwplot.imshow(rgb, doclf=1)
        top_dets = dets.compress(dets.scores > 0.2)
        top_dets.draw()


def demo_item():
    # xdoctest: +REQUIRES(env:DVC_DPATH)
    # This shows how you can use the dataloader to sample an arbitrary
    # spacetime volume.
    import watch
    import kwcoco
    dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    coco_fpath = dvc_dpath / 'Drop6/data_vali_split1.kwcoco.zip'
    coco_dset = kwcoco.CocoDataset(coco_fpath)
    ##'red|green|blue',
    from watch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    self = KWCocoVideoDataset(
        coco_dset,
        time_dims=7, window_dims=(1024, 1024),
        window_overlap=0,
        channels="(WV):red|green|blue",
        input_space_scale='0.3GSD',
        window_space_scale='0.3GSD',
        output_space_scale='0.3GSD',
        include_sensors='WV',
        #normalize_peritem='nir',
        dist_weights=0,
        quality_threshold=0,
        neg_to_pos_ratio=0, time_sampling='soft2',
    )
    self.requested_tasks['change'] = 1
    self.requested_tasks['saliency'] = 1
    self.requested_tasks['class'] = 0
    self.requested_tasks['boxes'] = 1
    item = self[0]

    # target = {
    #     'main_idx': 16,
    #     'video_id': 1,
    #     'gids': [171, 112, 172, 154, 331, 128, 424],
    #     'space_slice': (slice(58, 143, None), slice(411, 496, None)),
    #     'scale': 3.0303030303030303,
    #     'fliprot_params': {'rot_k': 1, 'flip_axis': None},
    #     'allow_augment': False}
    # item = self[target]

    target = {'main_idx': 4,
              'video_id': 2,
              'gids': [769, 498, 497, 713, 522, 562, 764],
              'space_slice': (slice(236, 267, None), slice(602, 633, None)),
              'scale': 33.333333333333336,
              'fliprot_params': {'rot_k': 1, 'flip_axis': (0,)},
              'allow_augment': False}
    item = self[target]
    return item


def _devtest():
    item = demo_item()

    import kwplot
    kwplot.plt.ion()
    kwplot.imshow(normed_rgb, doclf=1)
    top_dets = dets.compress(dets.scores > 0.2)
    top_dets.draw()

    # kwplot.imshow(normed_item)
    # canvas = self.draw_item(item)
    # import kwplot
    # kwplot.plt.ion()
    # kwplot.imshow(canvas)



if __name__ == '__main__':
    """

    CommandLine:
        xdoctest -m watch.tasks.building_detector.predict
    """
    main()
