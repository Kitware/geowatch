import torch
import kwcoco
import os
import json
import tifffile
from .time_sort_module import time_sort


def extract_features(checkpoint,
                     data_folder,
                     kwcoco_file,
                     output_kwcoco,
                     output_folder,
                     image_ids,
                     sensor,
                     panchromatic,
                     device='cuda'):
    """
    Function for extracting features given kwcoco reference to data and
    annotations. Output is copy of input kwcoco file with path towards saved
    pixel-wise features for image_id x saved under
    dset.imgs[x]['time_sort_features'].

    Args:
        checkpoint:  Path to checkpoint of lightning module. Default is UNet base trained on image sorting into before/after.
        data_folder:  Path to dvc repo
        kwcoco_file:  Path to kwcoco file with data annotations
        output_kwcoco:  Destination of output kwcoco file. set to same path as kwcoco_file to simply add paths to feature tensors to the existing file
        output_folder:  destination for feature tensors, stored as .pt files
        image_ids:  Set of image ids (corresponding to image ids in kwcoco_file) from which to extract features. image_ids from non-specified sensors will be skipped. Set to 0 to include all available images.
        sensor:  Choose from S2, LC, or WV. Note: with default checkpoint, only S2 (3 channel) images can be processed
        panchromatic: Set to True to return panchromatic (single channel) WV images where applicable. Otherwise 8 channel images will be returned.

    """

    extractor = time_sort.load_from_checkpoint(
        checkpoint, map_location='cuda').to(device)

    #  dataset = kwcoco.CocoDataset(kwcoco)
    with open(kwcoco_file) as read:
        dataset = json.load(read)

    if not os.path.exists(output_kwcoco):
        with open(output_kwcoco, 'w') as new_file:
            json.dump(dataset, new_file)

    dataset = kwcoco.CocoDataset(output_kwcoco)

    if not image_ids:
        # include all available images if none are specified
        image_ids = range(1, 1 + len(dataset.imgs))

    sensor_list = dataset.images().lookup('sensor_coarse', keepid=True)
    sensor_ids = [ID for ID in sensor_list if sensor_list[ID] == sensor]

    for x in image_ids:
        if x in sensor_ids:
            print('Processing image {}'.format(x))
            file_name = dataset.imgs[x]['file_name']

            directory, _ = os.path.split(file_name)

            if not os.path.exists(os.path.join(output_folder, directory)):
                os.makedirs(os.path.join(output_folder, directory))

            image = torch.tensor(
                tifffile.imread(
                    os.path.join(
                        data_folder,
                        file_name)).astype('int32')).to(device).float()

            if len(image.shape) < 3:
                image = image.unsqueeze(-1)

            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)

            features, _, _, _ = extractor(image, image, 'x', 'x')
            save_name = file_name[:-4] + '.pt'
            torch.save(
                features.squeeze(),
                os.path.join(
                    output_folder,
                    save_name))
            dataset.imgs[x]['time_sort_features'] = os.path.join(
                output_folder, save_name)
            dataset.dump(dataset.fpath, newlines=True)
        else:
            print('Skipping image {}, sensor doesn\'t match'.format(x))


if __name__ == '__main__':
    # TODO: this should be broken out into a function.
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        default='logs/drop0_sort/train_vidio_1/2021-05-14/default/version_7/checkpoints/epoch=97-step=195.ckpt')  # change to some shared place

    # drop0_aligned dataset arguments
    parser.add_argument(
        '--panchromatic',
        help='set flag for using panchromatic landsat imagery',
        action='store_true')
    parser.add_argument(
        '--sensor',
        type=str,
        help='Choose from WV, LC, or S2',
        default='S2')  # with default checkpoint, we must use RGB images
    parser.add_argument(
        '--data_folder',
        help='path to dvc on local machine',
        default='/localdisk0/SCRATCH/watch/smart_watch_dvc/drop0_aligned/')

    parser.add_argument(
        '--dataset',
        help='kwcoco file with dataset',
        default='/localdisk0/SCRATCH/watch/smart_watch_dvc/drop0_aligned/data.kwcoco.json')

    parser.add_argument(
        '--output_kwcoco',
        help='Filename to save output kwcoco file. Can replace old version.',
        default='/localdisk0/SCRATCH/watch/drop0_features/data_uky_time_sort_features.kwcoco.json')

    parser.add_argument(
        '--output_folder',
        help='Folder to store output feature tenors as .pt files',
        default='/u/eag-d1/scratch/ben/drop0_features/tensors')

    parser.add_argument(
        '--image_ids',
        nargs='+',
        type=int,
        help='Set to 0 for all available images. Otherwise take list of image ids for processing. Images from non-matching sensors will be automatically skipped.',
        default=0)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    extract_features(checkpoint=args.checkpoint,
                     data_folder=args.data_folder,
                     kwcoco_file=args.dataset,
                     output_kwcoco=args.output_kwcoco,
                     output_folder=args.output_folder,
                     image_ids=args.image_ids,
                     sensor=args.sensor,
                     panchromatic=args.panchromatic,
                     device=args.device
                     )
