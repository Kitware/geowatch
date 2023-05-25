import os
import argparse
from collections import defaultdict

import scipy
import torch
import kwcoco
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader

from watch.tasks.rutgers_material_seg_v2.matseg.models import build_model
from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_image import ImageStitcher_v2
from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_dataset import MATERIAL_TO_MATID
from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_mat_tran_mask import compute_material_transition_mask
from watch.tasks.rutgers_material_seg_v2.matseg.utils.utils_misc import load_cfg_file, generate_image_slice_object, create_hash_str

from watch.tasks.rutgers_material_seg_v2.mtm.dataset.inference_dataset import InferenceDataset
from watch.tasks.rutgers_material_seg_v2.mtm.utils.coco_stitcher import CocoStitchingManager, BlockingJobQueue


def make_material_predictions(eval_loader,
                              model,
                              output_coco_dset,
                              hash_name,
                              n_workers=4,
                              generate_mtm=True):
    """Generate and save material predictions to kwcoco file.

    Args:
        eval_loader (torch.utils.data.DataLoader): Dataset loader with region images to evaluate.
        model (torch.nn.Module): Material segmentation model.
        output_coco_dset (kwcoco.CocoDataset): The dataset where material predictions will be saved.
        hash_name (str): The hash name of the experiment.
        n_workers (int, optional): Number of threads to grab data. Defaults to 4.
        generate_mtm (bool, optional): Whether to generate material transition masks. Defaults to True.

    Returns:
        kwcoco.CocoDataset: Dataset with material predictions.
    """
    # Initialize model in case it hasnt been already.
    device = torch.device('cuda')
    model = model.to(device)
    model = model.eval()

    # Generate material predictions for each image in each region.
    pred_stitcher = ImageStitcher_v2('./', save_backend='gdal')
    pbar = tqdm(eval_loader, colour='green', desc='Generating material predictions')
    with torch.no_grad():
        for data in pbar:
            B = data['frame'].shape[0]

            # Convert images into model format.

            # Pass data into model.
            data['image'] = data['frame'].to(device)
            pred = model.forward(data)
            pred = pred.detach().cpu().numpy()

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

    # Finalize stitching operation and save images.
    stitched_predictions = pred_stitcher.get_combined_images()

    ## Sort by regions.
    region_predictions = defaultdict(list)
    for image_name, image in stitched_predictions.items():
        region_name = '_'.join(image_name.split('_')[:2])
        region_predictions[region_name].append(image)

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
        mtm_stitcher = CocoStitchingManager(output_coco_dset,
                                            short_code=f'mtm_{hash_name}',
                                            chan_code='mtm',
                                            stiching_space='video',
                                            writer_queue=writer_queue,
                                            expected_minmax=(0, 1))
    else:
        mtm_stitcher = None
    mat_pred_stitcher = CocoStitchingManager(output_coco_dset,
                                             short_code=f'materials_{hash_name}',
                                             chan_code='materials',
                                             stiching_space='video',
                                             writer_queue=writer_queue,
                                             expected_minmax=(0, 1))

    save_image_names = list(stitched_predictions.keys())
    for save_image_name in tqdm(save_image_names,
                                colour='green',
                                desc='Stitching and saving predictions'):
        mat_conf = stitched_predictions[save_image_name]
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

        del stitched_predictions[save_image_name]

    return output_coco_dset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kwcoco_fpath',
                        type=str,
                        help='Path to kwcoco file that we use to add our \
                        features to.')
    parser.add_argument('model_fpath',
                        type=str,
                        help='Path to material segmentation model that is \
                        used to generate material predictions as well as the material transition \
                        mask.')
    parser.add_argument('--config_fpath',
                        type=str,
                        default=None,
                        help='Path to the model`s configuration file.')
    parser.add_argument('--output_kwcoco_fpath',
                        type=str,
                        default=None,
                        help='Path to output kwcoco file.')
    parser.add_argument('--n_workers',
                        type=int,
                        default=None,
                        help='Number of parallel jobs to \
                        do data loading.')
    args = parser.parse_args()

    # Load configuration file.
    if args.config_fpath is None:
        ## Get experiment directory.
        exp_dir = '/'.join(args.model_fpath.split('/')[:-2])

        cfg_path = os.path.join(exp_dir, '.hydra', 'config.yaml')
    else:
        cfg_path = args.config_fpath

    if os.path.exists(cfg_path) is False:
        raise FileNotFoundError(f'Configuration file {cfg_path} does not exist.')

    cfg = load_cfg_file(cfg_path)

    ## Overwrite specific configuration file attributes.
    if args.n_workers is not None:
        cfg.n_workers = args.n_workers

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
                        checkpoint_path=args.model_fpath)

    # Create a new kwcoco file to store predictions
    og_kwcoco_dset = kwcoco.CocoDataset(args.kwcoco_fpath)
    output_coco_dset = og_kwcoco_dset.copy()
    video_ids = list(output_coco_dset.videos())

    ## Release old dataset from memory.
    del og_kwcoco_dset

    ## Generate hash name for generation run.
    hash_name = create_hash_str(method_name='sha256', **vars(args))[:10]

    # Make predictions on the dataset video by video.
    # This is done to save on memory usage and can be made into parallel process.
    for video_id in tqdm(video_ids, desc='Regions to predict'):
        # Create a dataset.
        dataset = InferenceDataset(video_id,
                                   channels=cfg.dataset.channels,
                                   kwcoco_path=args.kwcoco_fpath,
                                   crop_params=crop_params)

        # Create a loader for this video.
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.n_workers,
        )

        # Create material predictions.
        output_coco_dset = make_material_predictions(loader, model, output_coco_dset, hash_name)

    # Generate where to save new kwcoco file.
    if args.output_kwcoco_fpath is None:
        save_path = args.kwcoco_fpath.replace('.kwcoco.zip', '_mat_preds.kwcoco.zip')
    else:
        save_path = args.output_kwcoco_fpath

    # Save kwcoco file with material features.
    print(f'Saving predictions to: \n {save_path}')
    output_coco_dset.dump(save_path)


if __name__ == '__main__':
    main()
