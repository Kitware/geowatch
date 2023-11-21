import torch
import kwcoco
import os
import ubelt as ub
import kwimage
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

    input_dset = kwcoco.CocoDataset.coerce(kwcoco_file)
    output_dset = input_dset.copy()

    if not image_ids:
        # include all available images if none are specified
        image_ids = list(output_dset.index.imgs.keys())

    # Only take images that match the requested sensor
    valid_images = output_dset.images(image_ids)
    flags = [sensor == _ for _ in valid_images.lookup('sensor_coarse')]
    image_ids = valid_images.compress(flags)

    # TODO: could add a subdirectory using some tag associated with the
    # model to differentiate between features from different trained models
    os.makedirs(output_folder, exist_ok=True)

    # TODO: prediction would be faster with a dataset that loaded images
    # in the background while the GPU was predicting.

    # TODO: prediction will likely need to be done on a sliding window

    for gid in ub.ProgIter(image_ids, 'Process image'):

        img = output_dset.index.imgs[gid]
        # The image name should be unique, but if it does not exist, then
        # we have to get creative
        name = img.get('name', None)
        if name is None:
            name = 'timefeat_{:06d}'.format(gid)
        # Construct the filepath we will save the features to
        feature_fpath = os.path.join(output_folder, name + '.tif')

        # TODO: ensure the correct channels and scale wrt to the model are used
        delayed_image = output_dset.delayed_load(gid)
        im = delayed_image.finalize()

        # TODO: Ensure normalization is the same as in training
        # This should be accomplished by storing that info with the model
        image = torch.from_numpy(im.astype('float32')).to(device)

        if len(image.shape) < 3:
            image = image.unsqueeze(-1)

        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)

        batch_features, _, _, _ = extractor(image, image, 'x', 'x')

        # Assume batch size of 1
        item_features = batch_features[0]

        item_features_np = item_features.data.cpu().numpy().transpose(1, 2, 0)

        height, width, num_bands = item_features_np.shape

        # The input to the network is in "video-space", and the output is given
        # in the same "video-space" space. The output is going to be added as a
        # new auxiliary channel(s) to the image, so we need to specify the warp
        # from auxiliary space to image space, because auxiliary space in this
        # case is video space, we can use the inverse of the image-to-video
        # transform in the image dictionary.
        warp_img_to_vid = kwimage.Affine.coerce(img.get('warp_img_to_vid', None))
        warp_aux_to_img = warp_img_to_vid.inv()

        # TODO: need to come up with a channel code to represent this.
        # currently this could be done by any random 64 codes separated by
        # pipes but we may want to update kwcoco to be nicer in the way
        # it handles larger numbers of channels
        quick_chan_codes = ['UKy{:02d}'.format(i) for i in range(num_bands)]
        channels = '|'.join(quick_chan_codes)

        # Write the data to disk
        kwimage.imwrite(feature_fpath, item_features_np, backend='gdal', space=None)

        # Register the data in the output kwcoco manifest
        _temp_add_auxiliary(output_dset, gid, feature_fpath, width, height,
                            warp_aux_to_img, channels, num_bands)

    output_dset.fpath = output_kwcoco
    print('Write to output_dset.fpath = {!r}'.format(output_dset.fpath))
    output_dset.dump(output_dset.fpath, newlines=True)


def _temp_add_auxiliary(self, gid, fpath, width, height, warp_aux_to_img, channels, num_bands):
    """
    Adds an auxiliary file to an image.

    Temporary function while the kwcoco API is finalized
    """
    aux = {
        'file_name': fpath,
        'width': width,
        'height': height,
        'warp_aux_to_img': kwimage.Affine.coerce(warp_aux_to_img).concise(),
        'channels': channels,
        'num_bands': num_bands,
    }
    # lookup the image you want to add to
    img = self.index.imgs[gid]
    # Ensure there is an auxiliary image list
    auxiliary = img.setdefault('auxiliary', [])
    # Add the auxiliary information to the image
    auxiliary.append(aux)
    self._invalidate_hashid()


def main():
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


if __name__ == '__main__':
    """
    CommandLine:
        WATCH_DATA_DPATH=$(geowatch_dvc)

    """
    main()
