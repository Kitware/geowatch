import kwimage
import kwarray
import torch
import ubelt as ub
from argparse import ArgumentParser, RawTextHelpFormatter
from tqdm import tqdm
import os
# local imports
from .pretext_model import pretext
from .data.datasets import gridded_dataset
from watch.utils.lightning_ext import util_globals
from .segmentation_model import segmentation_model as seg_model
from watch.utils import util_kwimage


class predict(object):
    def __init__(self, args):
        ###
        self.finalized_gids = set()
        self.stitcher_dict = {}
        if 'all' in args.tasks:
            self.tasks = ['segmentation', 'before_after', 'pretext']
        else:
            self.tasks = args.tasks
        ### Define tasks
        if 'segmentation' in self.tasks:
            segmentation_model = seg_model.load_from_checkpoint(args.segmentation_ckpt_path, dataset=None)
            segmentation_model = segmentation_model.to(args.device)

        if 'pretext' in self.tasks:
            pretext_model = pretext.load_from_checkpoint(args.pretext_ckpt_path, train_dataset=None, vali_dataset=None)
            pretext_model = pretext_model.eval().to(args.device)
            # pretext_hparams = pretext_model.hparams

        self.in_feature_dims = pretext_model.hparams.feature_dim_shared
        if args.do_pca:
            self.pca_projector = torch.load(args.pca_projection_path)
            self.out_feature_dims = self.pca_projector.shape[0]
        else:
            self.out_feature_dims = self.in_feature_dims

        self.num_out_channels = self.out_feature_dims
        if 'segmentation' in self.tasks:
            self.num_out_channels +=1
        if 'before_after' in self.tasks:
            self.num_out_channels +=1

        self.save_channels = f'invariants:{self.num_out_channels}'
        self.output_kwcoco_path = args.output_kwcoco
        out_folder, _ = os.path.split(self.output_kwcoco_path)
        self.output_feat_dpath = os.path.join(out_folder, 'uky_invariants')
        self.imwrite_kw = {
        'compress': 'DEFLATE',
        'backend': 'gdal',
        'blocksize': 64,
        }

    def finalize_image(self, gid):
        self.finalized_gids.add(gid)
        stitcher = self.stitcher_dict[gid]
        recon = stitcher.finalize()
        self.stitcher_dict.pop(gid)

        save_path = os.path.join(self.output_feat_dpath, f'invariants_{gid}.tif')
        save_path = os.fspath(save_path)
        kwimage.imwrite(save_path, recon, backend='gdal', space=None,
                        **self.imwrite_kw)

        aux_height, aux_width = recon.shape[0:2]
        img = self.output_coco_dataset.index.imgs[gid]
        warp_aux_to_img = kwimage.Affine.scale(
            (img['width'] / aux_width,
             img['height'] / aux_height))

        aux = {
            'file_name': save_path,
            'height': aux_height,
            'width': aux_width,
            'channels': self.save_channels,
            'warp_aux_to_img': warp_aux_to_img.concise(),
        }
        auxiliary = img.setdefault('auxiliary', [])
        auxiliary.append(aux)

    def forward(self, args):
        try:
            device = int(args.device)
        except Exception:
            device = args.device
        num_workers = util_globals.coerce_num_workers(args.num_workers)
        print('num_workers = {!r}'.format(num_workers))

        dataset = gridded_dataset(args.input_kwcoco, args.bands, mode='test')

        loader = torch.utils.data.DataLoader(
            dataset, num_workers=num_workers, batch_size=1, shuffle=False)
        num_batches = len(loader)

        # Start background processes
        # Build a task queue for background write results workers (Not currently using this)
        # queue = util_parallel.BlockingJobQueue(max_workers=0)
        from watch.utils import util_parallel
        write_workers = util_globals.coerce_num_workers(args.write_workers)
        writer = util_parallel.BlockingJobQueue(max_workers=write_workers)

        output_dset = dataset.dset.copy()
        output_dset.reroot(absolute=True)  # Make all paths absolute
        output_dset.fpath = args.output_kwcoco  # Change output file path and bundle path
        output_dset.reroot(absolute=False)  # Reroot in the new bundle path

        bundle_dpath = ub.Path(output_dset.bundle_dpath)

        # save_dpath = (bundle_dpath / 'uky_invariants').ensuredir()

        self.imwrite_kw = {
            'compress': 'DEFLATE',
            'backend': 'gdal',
            'blocksize': 64,
        }

        print('Evaluating and saving features')

        with torch.set_grad_enabled(False):
            seen_images = set()

            for idx, batch in tqdm(enumerate(loader), total=num_batches, desc='Compute features'):
                save_feat = []
                if 'pretext' in args.tasks:
                    image_stack = torch.stack([batch['image1'], batch['image2'], batch['offset_image1'], batch['augmented_image1']], dim=1)
                    image_stack = image_stack.to(device)

                    #select features corresponding to first image
                    features = self.pretext_model(image_stack)[:, 0, :, :, :]

                    if args.do_pca:
                        features = torch.einsum('xy,byhw->bxhw', self.pca_projector, features)

                    save_feat.append(features.squeeze().permute(1, 2, 0).cpu().numpy())

                if 'before_after' in args.tasks:
                    ### TO DO: Set to output of separate model.
                    before_after_heatmap = self.pretext_model.shared_step(batch)['before_after_heatmap'][0].permute(1, 2, 0)
                    before_after_heatmap = torch.sigmoid(before_after_heatmap[:, :, 1] - before_after_heatmap[:, :, 0]).unsqueeze(-1).cpu().numpy()
                    save_feat.append(before_after_heatmap)
                if 'segmentation' in args.tasks:
                    image_stack = [batch[key] for key in batch if key[:5] == 'image']
                    image_stack = torch.stack(image_stack, dim=1).to(args.device)
                    segmentation_heatmap = torch.sigmoid(self.segmentation_model(image_stack)['predictions'][0, 0, 1, :, :] - self.segmentation_model(image_stack)['predictions'][0, 0, 0, :, :]).unsqueeze(0).permute(1, 2, 0).cpu().numpy()
                    save_feat.append(segmentation_heatmap)

                save_feat = torch.cat(save_feat, dim=-1)
                # image_id = int(batch['img1_id'].item())
                # image_info = output_dset.index.imgs[image_id]
                # video_info = output_dset.index.videos[image_info['video_id']]

                # video_folder = (save_dpath / video_info['name']).ensuredir()

                # # Predictions are saved in 'video space', so warp_aux_to_img is the inverse of warp_img_to_vid
                # warp_img_to_vid = kwimage.Affine.coerce(image_info.get('warp_img_to_vid', None))
                # warp_aux_to_img = warp_img_to_vid.inv().concise()

                # # Get the output image dictionary to be added to
                # output_img = output_dset.index.imgs[image_id]

                tr = dataset.patches[idx]
                sample = dataset.sampler.load_sample(tr)
                tr = sample['tr']

                if len(current_gids) == 0:
                    current_gids = tr['gids'][0]
                else:
                    previous_gids = current_gids
                    current_gids = tr['gids']
                    mutually_exclusive = (set(previous_gids) - set(current_gids))
                    for gid in mutually_exclusive:
                        seen_images.add(gid)
                        writer.submit(self.finalize_image, gid)

                    for gid in current_gids:
                        if gid not in self.stitcher_dict.keys():
                            self.stitcher_dict[gid] = kwarray.Stitcher(
                                tr['space_dims'], device='numpy')
                        slice_ = tr['space_slice']
                        weights = util_kwimage.upweight_center_mask(save_feat.shape[0:2])[..., None]
                        self.stitcher_dict[gid].add(slice_, features, weight=weights)

                writer.wait_until_finished()

                for gid in self.stitcher_dict.keys():
                    writer.submit(self.finalize_image, gid)

        print('Write to dset.fpath = {!r}'.format(output_dset.fpath))
        output_dset.dump(output_dset.fpath, newlines=True)
        print('Done')


def main():
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    from scriptconfig.smartcast import smartcast
    parser.add_argument('--device', type=str, default='cuda')

    # pytorch lightning checkpoint
    parser.add_argument('--pretext_ckpt_path', type=str)
    parser.add_argument('--segmentation_ckpt_path', type=str)
    parser.add_argument('--before_after_ckpt_path', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', default=4, help='number of background data loading workers')
    parser.add_argument('--write_workers', default=0, help='number of background data writing workers')

    # data flags - make sure these match the trained checkpoint
    parser.add_argument('--sensor', type=smartcast, nargs='+', default=['S2', 'L8'])
    parser.add_argument('--bands', type=str, help='Choose bands on which to train. Can specify \'all\' for all bands from given sensor, or \'share\' to use common bands when using both S2 and L8 sensors', nargs='+', default=['shared'])
    # output flags
    parser.add_argument('--input_kwcoco', type=str, help='Path to kwcoco dataset with images to generate feature for', required=True)
    parser.add_argument('--output_kwcoco', type=str, help='Path to write an output kwcoco file. Output file will be a copy of input_kwcoco with addition feature fields generated by predict.py rerooted to point to the original data.', required=True)
    parser.add_argument('--tasks', nargs='+', help='Specify which tasks to choose from (segmentation, before_after, or pretext. Can also specify \'all\')', default=['all'])
    parser.add_argument('--do_pca', type=int, help='Set to 1 to perform pca. Choose output dimension in num_dim argument.', default=1)
    parser.add_argument('--pca_projection_path', type=str, help='Path to pca projection matrix', default='')

    parser.set_defaults(
        terminate_on_nan=True
        )

    args = parser.parse_args()

    if 'all' in args.tasks:
        args.tasks = ['segmentation', 'before_after', 'pretext']

    predict(args)


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.tasks.template.predict --help

        python -m watch.tasks.template.predict \
            --input_kwcoco=path/to/data.kwcoco.json
    """
    main()
