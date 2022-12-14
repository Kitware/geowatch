import os
import argparse

import torch
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader

from watch.tasks.rutgers_material_seg_v2.models import build_model
from watch.tasks.rutgers_material_seg_v2.utils.util_misc import load_cfg_file
from watch.tasks.rutgers_material_seg_v2.utils.util_image import ImageStitcher
from watch.tasks.rutgers_material_seg_v2.datasets import build_dataset, custom_collate_fn
from watch.tasks.rutgers_material_seg_v2.utils.util_misc import softmax, MATERIAL_TO_MATID


def generate_kwcoco_predictions(model, eval_loader, pred_save_path):
    device = torch.device('cuda')
    model = model.to(device)
    model = model.eval()

    # Get save dir to store prediction images.
    base_dir = '/'.join(pred_save_path.split('/')[:-1])
    exp_name = pred_save_path.split('/')[-1].split('.')[0]

    stitcher_save_dir = os.path.join(base_dir, '_material_predictions', exp_name)
    os.makedirs(stitcher_save_dir, exist_ok=True)

    pred_stitcher = ImageStitcher(stitcher_save_dir, save_backend='gdal')
    image_name_to_id = {}
    with torch.no_grad():
        for data in tqdm(eval_loader, colour='green', desc='Generating material predictions'):

            # Convert images into model format.
            # image_data = train_dataset.format_image_data(data['image'], data['buffer_mask'])
            b, c, h, w = data['image'].shape

            image_data = {}
            image_data['pixel_data'] = rearrange(data['image'], 'b c h w -> (b h w) c')
            image_data['pixel_data'] = image_data['pixel_data'].to(device)

            prediction = model.forward(image_data)

            # Resize to original crop size
            prediction = prediction.view(b, h, w, -1)
            prediction = prediction.permute(0, 3, 1, 2)

            # Convert predictions to numpy arrays.
            prediction = prediction.detach().cpu().numpy()

            # Add images to stitcher process.
            ## Get image names.
            image_names = [
                eval_loader.dataset.coco_dset.index.imgs[iid.item()]['name'] for iid in list(data['image_id'])
            ]

            # Save images from the stitcher object.
            for i in range(b):
                sm_pred = softmax(prediction[i], axis=0)
                pred_stitcher.add_image(sm_pred, image_names[i], data['crop_slice'][i], data['og_height'][i].item(),
                                        data['og_width'][i].item(), data['buffer_mask'][i])

                ## Additionally build mapping between image name and ids.
                if image_names[i] not in image_name_to_id.keys():
                    image_name_to_id[image_names[i]] = data['image_id'][i].item()

        # Finalize stitching operation and save images.
        save_paths, image_names, image_sizes = pred_stitcher.save_images()

        # Create a new kwcoco file.
        pred_coco_dset = eval_loader.dataset.coco_dset.copy()

        # Add material predictions to kwcoco file.
        for save_path, image_name, image_size in tqdm(zip(save_paths, image_names, image_sizes),
                                                      colour='green',
                                                      desc='Saving predictions'):
            # Get image ID.
            image_id = image_name_to_id[image_name]

            # Add material prediction to kwcoco file.
            pred_coco_dset.add_auxiliary_item(image_id,
                                              save_path,
                                              channels='mat_pred',
                                              width=image_size[-2],
                                              height=image_size[-1],
                                              warp_aux_to_img={'scale': 1.0})

        # Save kwcoco file.
        print('Dumping kwcoco file ... ')
        pred_coco_dset.dump(pred_save_path)
        print(f'Saved to: {pred_save_path}')

        return pred_save_path


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='Path to the model weights.')
    parser.add_argument('config_path',
                        type=str,
                        help='Path to the configuration file to recreate the model and correct dataset parameters.')
    parser.add_argument('target_dataset_path',
                        type=str,
                        help='Path of the input kwcoco dataset for model to make predictions on.')
    parser.add_argument('pred_save_path', type=str, help='Where the material predictions kwcoco file will be saved.')
    parser.add_argument('--include_labels',
                        default=False,
                        action='store_true',
                        help='Include material labels in the prediction script. Will be utilized for analysis scripts.')
    args = parser.parse_args()

    # Load configuration file.
    cfg = load_cfg_file(args.config_path)

    # Generate KWCOCO file with model predictions.

    ## Get evaluation dataset and loader.
    eval_dataset = build_dataset(cfg.dataset.name, args.target_dataset_path, 'eval', **cfg.dataset.kwargs)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=cfg.n_workers,
                             collate_fn=custom_collate_fn)

    ## Rebuild model accroding to configurations.
    n_material_classes = len(MATERIAL_TO_MATID.keys())
    model = build_model(cfg.model.name,
                        n_in_channels=eval_dataset.n_channels,
                        n_out_channels=n_material_classes,
                        **cfg.model.kwargs)

    ## Load model with best weights.
    model = model.load_from_checkpoint(args.checkpoint_path,
                                       n_in_channels=eval_dataset.n_channels,
                                       n_out_channels=n_material_classes)

    ## Generate kwcoco file with predictions.
    _ = generate_kwcoco_predictions(model, eval_loader, args.pred_save_path)


if __name__ == '__main__':
    predict()
