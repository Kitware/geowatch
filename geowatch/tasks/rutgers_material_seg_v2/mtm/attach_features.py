r"""
Prediction script for Rutgers material features which include intermediate material
features, material transition masks, and material predictions.

Given a checkout of the model weights, model config file and IARPA data, the following
demo computes and visualizes a subset of the features.

CommandLine:

    DVC_EXPT_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware=auto)
    DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware=auto)

    KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD
    RUTGERS_MATERIAL_MODEL_FPATH="$DVC_EXPT_DPATH/models/rutgers/ru_model_05_25_2023.ckpt"
    RUTGERS_MATERIAL_MODEL_CONFIG_FPATH="$DVC_EXPT_DPATH/models/rutgers/ru_config_05_25_2023.yaml"

    INPUT_DATASET_FPATH=$KWCOCO_BUNDLE_DPATH/imganns-KR_R001.kwcoco.zip
    OUTPUT_DATASET_FPATH=$KWCOCO_BUNDLE_DPATH/imganns-KR_R001_rutgers_test.kwcoco.zip

    echo "
    DVC_DATA_DPATH="$DVC_DATA_DPATH"
    DVC_EXPT_DPATH="$DVC_EXPT_DPATH"

    RUTGERS_MATERIAL_MODEL_FPATH="$RUTGERS_MATERIAL_MODEL_FPATH"
    RUTGERS_MATERIAL_MODEL_CONFIG_FPATH="$RUTGERS_MATERIAL_MODEL_CONFIG_FPATH"

    INPUT_DATASET_FPATH="$INPUT_DATASET_FPATH"
    OUTPUT_DATASET_FPATH="$OUTPUT_DATASET_FPATH"
    "

    cat "$RUTGERS_MATERIAL_MODEL_CONFIG_FPATH"
    python -m geowatch.utils.simple_dvc request $RUTGERS_MATERIAL_MODEL_FPATH
    kwcoco stats "$INPUT_DATASET_FPATH"

    export CUDA_VISIBLE_DEVICES="1"
    python -m geowatch.tasks.rutgers_material_seg_v2.predict \
        --kwcoco_fpath="$INPUT_DATASET_FPATH" \
        --model_fpath="$RUTGERS_MATERIAL_MODEL_FPATH" \
        --config_fpath="$RUTGERS_MATERIAL_MODEL_CONFIG_FPATH" \
        --output_kwcoco_fpath="$OUTPUT_DATASET_FPATH" \
        --num_workers=4

    geowatch stats $OUTPUT_DATASET_FPATH

    geowatch visualize $OUTPUT_DATASET_FPATH \
        --animate=True --channels="red|green|blue,mtm,materials.0:3,mat_feats.0:3,mat_feats.3:6" \
        --skip_missing=True --workers=4 --draw_anns=False --smart=True

    python -m geowatch.tasks.rutgers_material_seg_v2.visualize_material_features \
        $OUTPUT_DATASET_FPATH ./mat_visualize_test/


CommandLine:

    # For batch computation
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD
    python -m geowatch.cli.queue_cli.prepare_teamfeats \
        --base_fpath=$KWCOCO_BUNDLE_DPATH/imganns-*[0-9].kwcoco.zip \
        --expt_dvc_dpath="$DVC_EXPT_DPATH" \
        --with_materials=1 \
        --skip_existing=1 --run=1 \
        --assets_dname=teamfeats
        --gres=0, --tmux_workers=1 --backend=tmux --run=1
"""
import torch
import ubelt as ub
from tqdm import tqdm
import scriptconfig as scfg


class MaterialsPredictConfig(scfg.DataConfig):
    kwcoco_fpath = scfg.Value(None, required=True, help=ub.paragraph('''
                            KWCOCO file to add material predictions to.
                            '''))
    model_fpath = scfg.Value(None, required=True, help=ub.paragraph('''
                            Path to material segmentation model that is
                            used to generate material predictions as well
                            as the material transition mask.'''))
    config_fpath = scfg.Value(None, required=False, help=ub.paragraph('''
                            Path to the model`s configuration file.'''))
    output_kwcoco_fpath = scfg.Value(None, required=False, help=ub.paragraph('''
                            Path to output kwcoco file.'''))
    feature_layer = scfg.Value(1, required=False, help=ub.paragraph('''
                            Which feature layer to use. There are 6 output layers.
                            Default: 1 (2x upscale).
                            '''))
    n_feature_dims = scfg.Value(16, required=False, help=ub.paragraph('''
                            Number of feature dimensions to use in the current layer.
                            '''))
    workers = scfg.Value(None,
                          help=ub.paragraph('''Number of background data loading workers
                            '''),
                          alias=['num_workers'])
    assets_dname = scfg.Value('_assets', required=False, help=ub.paragraph('''
                            The name of the top-level directory to write new assets.
                            '''))
    include_sensors = scfg.Value(['S2', 'L8', 'WV'], required=False, help=ub.paragraph('''
                            Comma separated list of sensors to include.
                            '''))


__cli__ = MaterialsPredictConfig


def make_material_predictions(eval_loader,
                              model,
                              output_coco_dset,
                              hash_name,
                              n_workers=4,
                              generate_mtm=True,
                              feature_layer=1,
                              n_feature_dims=16,
                              asset_dname='_assets'):
    """Generate and save material predictions to kwcoco file.

    Args:
        eval_loader (torch.utils.data.DataLoader): Dataset loader with region images to
            evaluate.
        model (torch.nn.Module): Material segmentation model.
        output_coco_dset (kwcoco.CocoDataset): The dataset where material predictions
            will be saved.
        hash_name (str): The hash name of the experiment.
        n_workers (int, optional): Number of threads to grab data. Defaults to 4.
        generate_mtm (bool, optional): Whether to generate material transition masks.
            Defaults to True.
        feature_layer (int, optional): Which feature layer to use. Defaults to 1.
        n_feature_dims (int, optional): Number of feature dimensions to use in the
            current layer. Defaults to 16.
        asset_dname (str, optional): The name of the top-level directory to write new
            assets. Defaults to '_assets'.

    Returns:
        kwcoco.CocoDataset: Dataset with material predictions.
    """
    # Imports
    from collections import defaultdict

    import scipy
    from einops import rearrange

    from geowatch.utils.util_parallel import BlockingJobQueue
    from geowatch.tasks.fusion.coco_stitcher import CocoStitchingManager

    from geowatch.tasks.rutgers_material_seg_v2.matseg.utils.utils_image import ImageStitcher_v2
    from geowatch.tasks.rutgers_material_seg_v2.matseg.utils.utils_mat_tran_mask import compute_material_transition_mask

    # Initialize model in case it hasnt been already.
    device = torch.device('cuda')
    model = model.to(device)
    model = model.eval()

    # Initialize upsampler for features.
    upsampler = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    # Generate material predictions for each image in each region.
    pred_stitcher = ImageStitcher_v2('./', save_backend='gdal')
    feat_stitcher = ImageStitcher_v2('./', save_backend='gdal')
    pbar = tqdm(eval_loader, colour='green', desc='Generating material predictions')
    with torch.no_grad():
        for data in pbar:
            B = data['frame'].shape[0]

            # Convert images into model format.

            # Pass data into model.
            data['image'] = data['frame'].to(device)
            output = model.forward_feat(data)
            pred = output['logits']
            pred = pred.detach().cpu().numpy()

            mat_feats = output['enc_feats'][feature_layer][:, :n_feature_dims]
            mat_feats = upsampler(mat_feats)
            mat_feats = mat_feats.detach().cpu().numpy()

            for i in range(B):
                # Add the early and late predictions to the stitcher.
                sm_pred = scipy.special.softmax(pred[i], axis=0)  # type: ignore

                image_name = f"{data['region_name'][i]}_{data['image_id'][i]}"
                ex_crop_params = []
                for j in range(4):
                    ex_crop_params.append(data['crop_slice'][j][i].item())
                height = data['region_res'][0][i].item()
                width = data['region_res'][1][i].item()
                pred_stitcher.add_image(sm_pred, image_name, ex_crop_params, height, width)
                feat_stitcher.add_image(mat_feats[i], image_name, ex_crop_params, height, width)

    # Finalize stitching operation and save images.
    stitched_predictions = pred_stitcher.get_combined_images()
    stitched_features = feat_stitcher.get_combined_images()

    ## Sort by regions.
    region_predictions = defaultdict(list)
    for image_name, image in stitched_predictions.items():
        region_name = '_'.join(image_name.split('_')[:2])
        region_predictions[region_name].append(image)

    ## Sort by regions.
    region_features = defaultdict(list)
    for image_name, image in stitched_features.items():
        region_name = '_'.join(image_name.split('_')[:2])
        region_features[region_name].append(image)

    # Generate material transition matrix.
    mtm_region_preds = {}
    for region_name, region_preds in tqdm(region_predictions.items(),
                                          colour='red',
                                          desc='Generating MTM masks'):
        # Compute the material transition mask.
        mat_trans_mask, beg_mat_pred, end_mat_pred = compute_material_transition_mask(
            'hard_class_2',
            region_preds[0][None],
            region_preds[-1][None],
            heuristic='soften_seasonal')
        mtm_region_preds[region_name] = mat_trans_mask

    writer_queue = BlockingJobQueue(max_workers=n_workers)
    if generate_mtm:
        mtm_stitcher = CocoStitchingManager(
            output_coco_dset,
            short_code=f'materials/mtm_{hash_name}',
            chan_code='mtm',
            stiching_space='video',
            writer_queue=writer_queue,
            expected_minmax=(0, 1),
            assets_dname=asset_dname
        )
    else:
        mtm_stitcher = None
    mat_pred_stitcher = CocoStitchingManager(
        output_coco_dset,
        short_code=f'materials/materials_{hash_name}',
        chan_code='materials.0:9',
        stiching_space='video',
        writer_queue=writer_queue,
        expected_minmax=(0, 1),
        assets_dname=asset_dname
    )

    mat_feat_stitcher = CocoStitchingManager(
        output_coco_dset,
        short_code=f'materials/mat_feats_{hash_name}',
        chan_code=f'mat_feats.0:{n_feature_dims}',
        stiching_space='video',
        writer_queue=writer_queue,
        assets_dname=asset_dname
    )

    save_image_names = list(stitched_predictions.keys())
    for save_image_name in tqdm(save_image_names,
                                colour='green',
                                desc='Stitching and saving predictions'):
        mat_conf = stitched_predictions[save_image_name]
        region_feat_img = stitched_features[save_image_name]
        region_name = '_'.join(save_image_name.split('_')[:2])
        image_id = save_image_name.split('_')[-1]
        region_mtm = mtm_region_preds[region_name]
        h, w = region_mtm.shape
        gid = int(image_id)
        if mtm_stitcher:
            mtm_stitcher.accumulate_image(gid,
                                          space_slice=None,
                                          data=region_mtm,
                                          asset_dsize=(w, h))
            mtm_stitcher.submit_finalize_image(gid)
        mat_pred_stitcher.accumulate_image(gid,
                                           space_slice=None,
                                           data=rearrange(mat_conf, 'c h w -> h w c'),
                                           asset_dsize=(w, h))
        mat_pred_stitcher.submit_finalize_image(gid)

        mat_feat_stitcher.accumulate_image(gid,
                                           space_slice=None,
                                           data=rearrange(region_feat_img, 'c h w -> h w c'),
                                           asset_dsize=(w, h))
        mat_feat_stitcher.submit_finalize_image(gid)

        del stitched_predictions[save_image_name]
        del stitched_features[save_image_name]

    return output_coco_dset


def predict(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from geowatch.tasks.rutgers_material_seg_v2.predict import *  # NOQA
        >>> import kwcoco
        >>> import geowatch
        >>> dvc_data_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> dvc_expt_dpath = geowatch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> dset = kwcoco.CocoDataset(dvc_data_dpath / 'Drop6/imganns-KR_R001.kwcoco.zip')
        >>> deployed_weights_path = dvc_expt_dpath / 'models/rutgers/ru_model_05_25_2023.ckpt'
        >>> deployed_config_path = dvc_expt_dpath / 'models/rutgers/ru_model_05_25_2023.yaml'
        >>> kwargs = {
        >>>     'kwcoco_fpath': dset.fpath,
        >>>     'model_fpath': deployed_weights_path,
        >>>     'config_fpath': deployed_config_path,
        >>>     'output_kwcoco_fpath': ub.Path(dset.fpath).augment(stemsuffix='_rutgers', multidot=True),
        >>> }
        >>> cmdline = 0
        >>> predict(cmdline, **kwargs)
    """
    # Imports
    import os

    import rich
    import kwcoco
    from torch.utils.data import DataLoader

    from geowatch.tasks.rutgers_material_seg_v2.matseg.models import build_model
    from geowatch.tasks.rutgers_material_seg_v2.matseg.utils.utils_dataset import MATERIAL_TO_MATID
    from geowatch.tasks.rutgers_material_seg_v2.matseg.utils.utils_misc import load_cfg_file, generate_image_slice_object, create_hash_str
    from geowatch.tasks.rutgers_material_seg_v2.mtm.dataset.inference_dataset import InferenceDataset

    # Get config.
    script_config = MaterialsPredictConfig.cli(cmdline=cmdline, data=kwargs, strict=True)

    rich.print('config = {}'.format(ub.urepr(script_config, align=':', nl=1)))

    # Load configuration file.
    if script_config['config_fpath'] is None:
        ## Get experiment directory.
        exp_dir = '/'.join(script_config['model_fpath'].split('/')[:-2])

        cfg_path = os.path.join(exp_dir, '.hydra', 'config.yaml')
    else:
        cfg_path = script_config['config_fpath']

    if os.path.exists(cfg_path) is False:
        raise FileNotFoundError(f'Configuration file {cfg_path} does not exist.')

    cfg = load_cfg_file(cfg_path)

    ## Overwrite specific configuration file attributes.
    if script_config['workers'] is not None:
        cfg.n_workers = script_config['workers']

    # Create a dataset.
    ## Get the input crop resolution for this model.
    crop_params = generate_image_slice_object(cfg.crop_height, cfg.crop_width, cfg.crop_stride)

    # Load a model.
    if cfg.model.kwargs is None:
        cfg.model.kwargs = {}

    if hasattr(cfg.model, 'lr_scheduler_mode'):
        cfg.model.kwargs['lr_scheduler_mode'] = None

    if cfg.dataset.channels == 'RGB':
        n_channels = 3
    elif cfg.dataset.channels == 'RGB_NIR':
        n_channels = 4
    else:
        raise NotImplementedError

    model = build_model(None,
                        network_name=cfg.model.architecture,
                        encoder_name=cfg.model.encoder,
                        in_channels=n_channels,
                        out_channels=len(MATERIAL_TO_MATID.keys()),
                        loss_mode=cfg.model.loss_mode,
                        optimizer_mode=cfg.model.optimizer_mode,
                        class_weight_mode=cfg.model.class_weight_mode,
                        lr=cfg.model.lr,
                        wd=cfg.model.wd,
                        pretrain=cfg.model.pretrain,
                        to_rgb_fcn=None,
                        **cfg.model.kwargs,
                        checkpoint_path=script_config['model_fpath'])

    # Create a new kwcoco file to store predictions
    og_kwcoco_dset = kwcoco.CocoDataset(script_config['kwcoco_fpath'])
    output_coco_dset = og_kwcoco_dset.copy()
    video_ids = list(output_coco_dset.videos())

    ## Release old dataset from memory.
    del og_kwcoco_dset

    ## Generate hash name for generation run.
    hash_name = create_hash_str(method_name='sha256', **vars(script_config))[:10]

    # Make predictions on the dataset video by video.
    # This is done to save on memory usage and can be made into parallel process.
    for video_id in tqdm(video_ids, desc='Regions to predict'):
        # Create a dataset.
        dataset = InferenceDataset(video_id,
                                   channels=cfg.dataset.channels,
                                   kwcoco_path=script_config['kwcoco_fpath'],
                                   crop_params=crop_params,
                                   kwcoco_dset=output_coco_dset,
                                   sensors=script_config['include_sensors'])

        # Create a loader for this video.
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.n_workers,
        )

        # Create material predictions.
        output_coco_dset = make_material_predictions(loader,
                                                     model,
                                                     output_coco_dset,
                                                     hash_name,
                                                     feature_layer=script_config['feature_layer'],
                                                     n_feature_dims=script_config['n_feature_dims'],
                                                     asset_dname=script_config['assets_dname'])

    # Generate where to save new kwcoco file.
    if script_config['output_kwcoco_fpath'] is None:
        save_path = script_config['kwcoco_fpath'].replace('.kwcoco.zip', '_mat_preds.kwcoco.zip')
    else:
        save_path = script_config['output_kwcoco_fpath']

    # Save kwcoco file with material features.
    print(f'Saving predictions to: \n {save_path}')
    output_coco_dset.dump(save_path)


if __name__ == '__main__':
    predict()
