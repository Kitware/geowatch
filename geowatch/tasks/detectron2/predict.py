#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class DetectronPredictCLI(scfg.DataConfig):
    # TODO: scriptconfig Value classes should have tags for mlops pipelines
    # Something like tags ⊆ {in_path,out_path, algo_param, perf_param, primary, primary_in_path, primary_out_path}
    checkpoint_fpath = scfg.Value(None, help='path to the weights')
    model_fpath = scfg.Value(None, help='path to a model file: todo: bundle with weights')
    src_fpath = scfg.Value(None, help='input kwcoco file')
    dst_fpath = scfg.Value(None, help='output kwcoco file')
    write_heatmap = scfg.Value(False, help='if True, also write masks as heatmaps')
    nms_thresh = scfg.Value(0.0, help='nonmax supression threshold')
    workers = 4

    base = scfg.Value('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', help=ub.paragraph(
        '''
        detectron config that corresponds to the model. TODO: package and introspect this
        '''))

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = DetectronPredictCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        detectron_predict(config)

__cli__ = DetectronPredictCLI


def detectron_predict(config):
    import geowatch_tpl
    detectron2 = geowatch_tpl.import_submodule('detectron2')  # NOQA
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    import kwcoco
    import kwimage
    import kwarray
    import kwutil
    import torch
    import einops

    checkpoint_fpath = config.checkpoint_fpath
    src_fpath = config.src_fpath
    dst_fpath = config.dst_fpath

    proc_context = kwutil.ProcessContext(
        name='geowatch.tasks.detectron2.predict',
        config=kwutil.Json.ensure_serializable(dict(config)),
        track_emissions=True,
    )
    proc_context.start()
    print(f'proc_context.obj = {ub.urepr(proc_context.obj, nl=3)}')

    cfg = get_cfg()
    base_cfg = model_zoo.get_config_file(config.base)
    cfg.merge_from_file(base_cfg)

    # cfg.DATASETS.TRAIN = (dataset_infos['train']['name'],)
    # cfg.DATASETS.TEST = (dataset_infos['vali']['name'],)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')  # Let training initialize from model zoo
    # cfg.SOLVER.IMS_PER_BATCH = 2   # This is the real 'batch size' commonly known to deep learning people
    # cfg.SOLVER.BASE_LR = 0.00025   # pick a good LR
    # cfg.SOLVER.MAX_ITER = 120_000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    # cfg.SOLVER.STEPS = []          # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The 'RoIHead batch size'. 128 is faster, and good enough for this toy dataset (default: 512)
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 46  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
    cfg.MODEL.WEIGHTS = checkpoint_fpath

    # NEED to call detectron2 more efficiently
    predictor = DefaultPredictor(cfg)

    dset = kwcoco.CocoDataset(src_fpath)

    keep_annots = 0
    if not keep_annots:
        dset.clear_annotations()

    dset.reroot(absolute=True)
    dset.fpath

    # TODO: could be much more efficient
    torch_impl = kwarray.ArrayAPI.coerce('torch')()

    # import geowatch
    from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    dataset = KWCocoVideoDataset(
        dset,
        window_space_dims='full',
        channels='blue|green|red',
        time_dims=1,
        mode='test',
        # todo: enhance reduce item size to remove most information, but do
        # keep the image id.
        # reduce_item_size=True,
    )
    batch_item = dataset[0]
    dset.reroot(absolute=True)
    bundle_dpath = ub.Path(dst_fpath).parent.ensuredir()
    dset.fpath = dst_fpath

    # FIXME: We need a method to know what classes the detectron2 model was
    # trained with.
    classes = dset.object_categories()
    print(f'dset={dset}')

    if config.write_heatmap:
        from geowatch.tasks.fusion.coco_stitcher import CocoStitchingManager
        from kwutil import util_parallel
        writer_queue = util_parallel.BlockingJobQueue(
            mode='thread',
            # mode='serial',
            max_workers=2,
        )
        stitcher_common_kw = dict(
            # stiching_space='video',
            stiching_space='image',
            device='numpy',
            # thresh=config['thresh'],
            write_probs=True,
            # write_preds=config['write_preds'],
            # prob_compress=config['compress'],
            # prob_format=config['format'],
            # quantize=config['quantize'],
            expected_minmax=(0, 1),
            writer_queue=writer_queue,
            assets_dname='_assets',
            # memmap=config.memmap,
        )
        stitcher = CocoStitchingManager(
            dset,
            chan_code='salient',
            short_code='salient',
            num_bands=1,
            **stitcher_common_kw,
        )
        # if 0:
        #     head_stitcher.accumulate_image(
        #         gid, output_space_slice, probs,
        #         asset_dsize=output_image_dsize,
        #         scale_asset_from_stitchspace=scale_outspace_from_vid,
        #         weights=output_weights,
        #         downweight_edges=downweight_edges,
        #     )
        #     for gid in head_stitcher.ready_image_ids():
        #         head_stitcher._ready_gids.difference_update({gid})  # avoid race condition
        #         head_stitcher.submit_finalize_image(gid)
        #     print(f"Finalizing stitcher for {_head_key}")
        #     for gid in head_stitcher.managed_image_ids():
        #         head_stitcher.submit_finalize_image(gid)
    else:
        stitcher = None

    loader = dataset.make_loader(
        batch_size=1,
        num_workers=4,  # config.workers
    )
    # images = dset.images()
    import rich
    rich.print(f'Pred Dpath: [link={bundle_dpath}]{bundle_dpath}[/link]')

    batch_iter = ub.ProgIter(loader, total=len(loader), desc='predict')
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        for batch in batch_iter:
            # raise Exception
            assert len(batch) == 1
            batch_item = batch[0]
            image_id = batch_item['target']['gids'][0]
            im_chw = batch_item['frames'][0]['modes']['blue|green|red']
            im_hwc = einops.rearrange(im_chw, 'c h w -> h w c').numpy()

            # Run the model
            # outputs = predictor(im_hwc.numpy())
            height, width = im_chw.shape[1:3]
            image = predictor.aug.get_transform(im_hwc).apply_image(im_hwc)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(predictor.cfg.MODEL.DEVICE)

            inputs = {"image": image, "height": height, "width": width}
            outputs = predictor.model([inputs])[0]

            instances = outputs['instances']
            if len(instances):
                boxes = instances.pred_boxes
                scores = instances.scores
                boxes = kwimage.Boxes(boxes.tensor, format='ltrb').numpy()
                scores = torch_impl.numpy(instances.scores)
                # TODO: handle masks

                if hasattr(instances, 'pred_masks'):
                    pred_masks = torch_impl.numpy(instances.pred_masks)
                    segmentations = []
                    for cmask in pred_masks:
                        mask = kwimage.Mask.coerce(cmask)
                        poly = mask.to_multi_polygon()
                        segmentations.append(poly)
                else:
                    segmentations = None

                pred_class_indexes = torch_impl.numpy(instances.pred_classes)

                dets = kwimage.Detections(
                    boxes=boxes,
                    scores=scores,
                    class_idxs=pred_class_indexes,
                    segmentations=segmentations,
                    classes=classes,
                )
                if config.nms_thresh and config.nms_thresh > 0:
                    dets = dets.non_max_supress(thresh=config.nms_thresh)

                for ann in dets.to_coco(style='new'):
                    ann['image_id'] = image_id
                    catname = ann.pop('category_name')
                    ann['category_id'] = dset.ensure_category(catname)
                    ann['role'] = 'prediction'
                    dset.add_annotation(**ann)
            else:
                dets = kwimage.Detections.random(0)
                dets.data['segmentations'] = []

            if stitcher is not None:
                frame_info = batch_item['frames'][0]
                output_image_dsize = frame_info['output_image_dsize']
                output_space_slice = frame_info['output_space_slice']
                scale_outspace_from_vid = frame_info['scale_outspace_from_vid']

                import numpy as np
                sorted_dets = dets.take(dets.scores.argsort())
                probs = np.zeros(output_image_dsize[::-1], dtype=np.float32)
                for sseg, score in zip(sorted_dets.data['segmentations'], sorted_dets.scores):
                    sseg.data.fill(probs, value=float(score), assert_inplace=True)

                stitcher.accumulate_image(
                    image_id, output_space_slice, probs,
                    asset_dsize=output_image_dsize,
                    scale_asset_from_stitchspace=scale_outspace_from_vid,
                    # weights=output_weights,
                    # downweight_edges=downweight_edges,
                )
                # hack / fixme: this is ok, when batches correspond with
                # images but not if we start to window.
                stitcher.submit_finalize_image(image_id)

    if stitcher is not None:
        writer_queue.wait_until_finished()  # hack to avoid race condition
        # Prediction is completed, finalize all remaining images.
        print(f"Finalizing stitcher for {stitcher}")
        for gid in stitcher.managed_image_ids():
            stitcher.submit_finalize_image(gid)
        writer_queue.wait_until_finished()

    dset.dataset.setdefault('info', [])
    proc_context.stop()
    print(f'proc_context.obj = {ub.urepr(proc_context.obj, nl=3)}')
    dset.dataset['info'].append(proc_context.obj)
    dset.dump()
    import rich
    rich.print(f'Wrote to: [link={bundle_dpath}]{bundle_dpath}[/link]')


if __name__ == '__main__':
    """

    CommandLine:
        python -m geowatch.tasks.detectron2.predict
    """
    __cli__.main()
