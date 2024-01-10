#!/usr/bin/env python3
r"""
Prediction script for Rutgers Material Semenatic Segmentation Models

CommandLine:

    export CUDA_VISIBLE_DEVICES=1
    DVC_DPATH=$(geowatch_dvc)

    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10
    BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json

    RUTGERS_MATERIAL_MODEL_FPATH="$DVC_DPATH/models/rutgers/rutgers_peri_materials_v3/experiments_epoch_18_loss_59.014100193977356_valmF1_0.18694573888313187_valChangeF1_0.0_time_2022-02-01-01:53:20.pth"

    RUTGERS_MATERIAL_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/rutgers_material_seg_v3.kwcoco.json

    # Generate Rutgers Features
    python -m geowatch.tasks.rutgers_material_seg.predict \
        --test_dataset=$BASE_COCO_FPATH \
        --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
        --default_config_key=iarpa \
        --pred_dataset=$RUTGERS_MATERIAL_COCO_FPATH \
        --num_workers=avail \
        --batch_size=4 --devices auto:1
        --export_raw_features True

"""
import os
import torch
import datetime
import random
import kwcoco
import kwimage
import kwarray
import ndsampler
import numpy as np
from tqdm import tqdm  # NOQA
import ubelt as ub
import pathlib
import geowatch.tasks.rutgers_material_seg.utils.utils as utils
from kwutil import util_parallel
from geowatch.tasks.rutgers_material_seg.models import build_model
from geowatch.tasks.rutgers_material_seg.datasets.iarpa_contrastive_dataset import SequenceDataset
import torch.nn.functional as F

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class Evaluator(object):
    """
    CommandLine:
        DVC_DPATH=$(geowatch_dvc)
        DVC_DPATH=$DVC_DPATH xdoctest -m geowatch.tasks.rutgers_material_seg.predict Evaluator

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from geowatch.tasks.rutgers_material_seg.predict import *  # NOQA
        >>> import geowatch
        >>> dvc_dpath = geowatch.find_dvc_dpath()
        >>> checkpoint_fpath = dvc_dpath / 'models/rutgers/rutgers_peri_materials_v3/experiments_epoch_18_loss_59.014100193977356_valmF1_0.18694573888313187_valChangeF1_0.0_time_2022-02-01-01:53:20.pth'
        >>> #checkpoint_fpath = dvc_dpath / 'models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_2021101T16277.pth'
        >>> #  Write out smaller version of the dataset
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset(dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data_nowv_vali.kwcoco.json')
        >>> images = dset.videos(names=['KR_R001']).images[0]
        >>> sub_images = images.compress([s != 'WV' for s in images.lookup('sensor_coarse')])[::100]
        >>> sub_dset = dset.subset(sub_images)
        >>> sub_dset.fpath = (dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/small_test_data_nowv_vali.kwcoco.json')
        >>> sub_dset.dump(sub_dset.fpath)
        >>> input_kwcoco = sub_dset.fpath
        >>> #
        >>> src_coco_fpath = sub_dset.fpath
        >>> dst_coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/mat_test.kwcoco.json'
        >>> kwargs = dict(
        >>>     default_config_key='iarpa',
        >>>     checkpoint_fpath=checkpoint_fpath,
        >>>     test_dataset=src_coco_fpath,
        >>>     pred_dataset=dst_coco_fpath,
        >>>     feat_dpath=dst_coco_fpath.parent / '_assets/test_rutgers_material_seg',
        >>>     devices='auto:1',
        >>>     num_workers='avail',
        >>>     save_raw_features=1,
        >>>     batch_size=32,
        >>>     cache=0,
        >>>     #num_workers=0,
        >>>     #gpu=None,
        >>> )
        >>> cmdline = False
        >>> evaler = build_evaler(cmdline=cmdline, **kwargs)
        >>> self = evaler
        >>> evaler.forward()
    """

    def __init__(self,
                 model: object,
                 checkpoint_fpath: str,
                 input_coco_dset: SequenceDataset,
                 output_coco_dataset: kwcoco.CocoDataset,
                 write_probs : bool = True,
                 device=None,
                 config : dict = None,
                 output_feat_dpath : pathlib.Path = None,
                 batch_size=1,
                 num_workers=0,
                 imwrite_kw={},
                 cache: bool = False,
                 save_raw_features: bool = False):
        """
        Evaluator class

        Args:
            model (object): trained or untrained model
            eval_loader (SequenceDataset): dataset with evaluation data
            optimizer (object): optimizer to train with
            scheduler (object): scheduler to train with
            cache (bool): if True, will only predict on images that do
                not have a feature computed for them yet.
        """

        self.model = model
        self.checkpoint_fpath = checkpoint_fpath
        self.save_raw_features = save_raw_features
        self.num_workers = num_workers
        self.output_coco_dataset = output_coco_dataset
        self.write_probs = write_probs
        self.device = device
        self.config = config
        self.num_classes = self.config['data']['num_classes']
        self.output_feat_dpath = output_feat_dpath
        self.stitcher_dict = {}
        if self.save_raw_features:
            # self.stitcher_dict_up3 = {}
            self.stitcher_dict_up5 = {}
        self.finalized_gids = set()
        self.imwrite_kw = imwrite_kw
        self.cache = cache
        self.batch_size = batch_size

        self.input_coco_dset = input_coco_dset

        # Hack together a channel code
        self.chan_code = '|'.join(['matseg_{}'.format(i) for i in range(self.num_classes)])
        # self.chan_code = '|'.join(['matseg.{}'.format(i) for i in range(self.num_classes)])

        self.output_channels = kwcoco.FusedChannelSpec.coerce(self.chan_code).concise()
        self.concise_chan_code = self.output_channels.spec
        self.concise_chan_path_code = self.output_channels.path_sanitize()

    def _output_path_for_image(self, gid):
        img = self.output_coco_dataset.index.imgs[gid]
        img_name = img.get('name', f'gid{gid:08d}')
        save_path = self.output_feat_dpath / f'{img_name}_{self.concise_chan_path_code}.tif'
        return save_path

    def _features_path_for_image(self, gid, layer):
        img = self.output_coco_dataset.index.imgs[gid]
        img_name = img.get('name', f'gid{gid:08d}')
        # FIXME: this shouldn't use the concise channel code, this should
        # use the same name as the path-sanatized channel string.
        save_path = self.output_feat_dpath / f'{img_name}_{self.concise_chan_path_code}_{layer}.tif'
        return save_path

    @profile
    def finalize_image(self, gid, cached=False):
        """
        Finializes the probabilities accumulated in the stitcher for a
        particular image, saves it to disk, and updates the kwcoco file in
        memory.

        Args:
            gid (int): image id to finalize
            cached (bool): if True, only updates the kwcoco file, assumes
                the file exists on disk.
        """
        img = self.output_coco_dataset.index.imgs[gid]

        self.finalized_gids.add(gid)

        if not cached:
            stitcher = self.stitcher_dict[gid]
            if self.save_raw_features:
                # stitcher_up3 = self.stitcher_dict_up3[gid]
                stitcher_up5 = self.stitcher_dict_up5[gid]

            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'invalid value encountered in true_divide', category=RuntimeWarning)
                recon = stitcher.finalize()

                if self.save_raw_features:
                    # recon_up3 = stitcher_up3.finalize()
                    recon_up5 = stitcher_up5.finalize()
                    # self.stitcher_dict_up3.pop(gid)
                    self.stitcher_dict_up5.pop(gid)
            self.stitcher_dict.pop(gid)
        else:
            recon = None
            if self.save_raw_features:
                # recon_up3 = None
                recon_up5 = None

        from geowatch.tasks.fusion.predict import quantize_float01
        # Note using -11 and +11 as a tradeoff range because we cannot
        # guarentee the bounds of this data. Usually it is mean zero with
        # a std < 3, so this should be a decent range to work within.
        quant_recon, quantization = quantize_float01(recon, old_min=-11, old_max=11)
        nodata = quantization['nodata']

        save_path = self._output_path_for_image(gid)
        save_path = os.fspath(save_path)

        if self.save_raw_features:
            # quant_recon_up3, quantization_up3 = quantize_float01(recon_up3, old_min=-11, old_max=11)
            # nodata_up3 = quantization_up3['nodata']
            # save_path_up3 = self._features_path_for_image(gid, 'up3')
            # save_path_up3 = os.fspath(save_path_up3)

            quant_recon_up5, quantization_up5 = quantize_float01(recon_up5, old_min=-11, old_max=11)
            nodata_up5 = quantization_up5['nodata']
            save_path_up5 = self._features_path_for_image(gid, 'up5')
            save_path_up5 = os.fspath(save_path_up5)

        if cached:
            aux_height, aux_width = kwimage.load_image_shape(save_path)[0:2]

            if self.save_raw_features:
                # aux_height_up3, aux_width_up3 = kwimage.load_image_shape(save_path_up3)[0:2]
                aux_height_up5, aux_width_up5 = kwimage.load_image_shape(save_path_up5)[0:2]
        else:
            kwimage.imwrite(save_path, quant_recon, backend='gdal', space=None,
                            nodata=nodata, **self.imwrite_kw)
            aux_height, aux_width = quant_recon.shape[0:2]

            if self.save_raw_features:
                # kwimage.imwrite(save_path_up3, quant_recon_up3, backend='gdal', space=None,
                #                 nodata=nodata_up3, **self.imwrite_kw)
                # aux_height_up3, aux_width_up3 = quant_recon_up3.shape[0:2]

                kwimage.imwrite(save_path_up5, quant_recon_up5, backend='gdal', space=None,
                                nodata=nodata_up5, **self.imwrite_kw)
                aux_height_up5, aux_width_up5 = quant_recon_up5.shape[0:2]

        warp_aux_to_img = kwimage.Affine.scale(
            (img['width'] / aux_width,
             img['height'] / aux_height))

        aux = {
            'file_name': save_path,
            'height': aux_height,
            'width': aux_width,
            'channels': self.concise_chan_code,
            'warp_aux_to_img': warp_aux_to_img.concise(),
            'quantization': quantization,
        }

        auxiliary = img.setdefault('auxiliary', [])
        auxiliary.append(aux)

        if self.save_raw_features:
            # aux_up3 = {
            #     'file_name': save_path_up3,
            #     'height': aux_height,
            #     'width': aux_width,
            #     'channels': 'mat_up3:128',
            #     'warp_aux_to_img': warp_aux_to_img.concise(),
            #     'quantization': quantization_up3,
            # }
            # auxiliary.append(aux_up3)

            aux_up5 = {
                'file_name': save_path_up5,
                'height': aux_height,
                'width': aux_width,
                'channels': 'mat_up5:64',
                'warp_aux_to_img': warp_aux_to_img.concise(),
                'quantization': quantization_up5,
            }

            auxiliary.append(aux_up5)

    @profile
    def eval(self) -> tuple:
        """
        Execute predictions on the evaluator kwcoco file.

        Dumps images and the final kwcoco file to disk.
        """
        # from geowatch.utils import util_kwimage
        current_gids = []
        previous_gids = []

        config = self.config
        window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size'])  # [t,h,w]
        input_dims = (config['data']['image_size'], config['data']['image_size'])
        channels = config['data']['channels']

        if self.cache:
            # Detect if we already wrote predictions for some images.
            miss_gids = []
            hit_gids = []
            all_gids = list(self.input_coco_dset.index.imgs.keys())
            for gid in all_gids:

                fpaths = []
                fpath = self._output_path_for_image(gid)
                fpaths.append(fpath)
                if self.save_raw_features:
                    save_path_up5 = self._features_path_for_image(gid, 'up5')
                    # save_path_up3 = self._features_path_for_image(gid, 'up3')
                    fpaths.append(save_path_up5)
                    # fpaths.append(save_path_up3)

                if all(p.exists() for p in fpaths):
                    hit_gids.append(gid)
                else:
                    miss_gids.append(gid)
            # NOTE: WE ARE ASSUMING ONLY 1 TIME STEP IS USED
            # THIS HAS TO CHANGE IF MULTIPLE TIMESTEPS ARE USED
            subset_input_coco = self.input_coco_dset.subset(miss_gids)
            print(f'Found {len(hit_gids)} / {len(all_gids)} cached features')
        else:
            subset_input_coco = self.input_coco_dset

        # Build the torch datasets / loaders
        sampler = ndsampler.CocoSampler(subset_input_coco)
        window_overlap = 0.3  # TODO parametarize
        eval_dataset = SequenceDataset(
            sampler, window_dims, input_dims, channels, training=False,
            window_overlap=window_overlap, inference_only=True)
        print(f'{len(eval_dataset)=}')

        if len(eval_dataset) == 0:
            # hack for case where everything is already cached
            eval_dataloader = []
        else:
            eval_dataloader = eval_dataset.make_loader(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )

        dataloader_iter = iter(eval_dataloader)
        writer = util_parallel.BlockingJobQueue(max_workers=self.num_workers)

        seen = set()

        if len(eval_dataloader):
            print('read self.checkpoint_fpath = {!r}'.format(self.checkpoint_fpath))
            checkpoint_state = torch.load(self.checkpoint_fpath)
            self.model.load_state_dict(checkpoint_state['model'])
            print(f"loadded model weights from: {self.checkpoint_fpath}")
            # print(f"Missing keys from loaded model: {missing_keys}, unexpected keys: {unexpexted_keys}")
            self.model = self.model.to(self.device)
            self.model.eval()

        with torch.no_grad():
            # from functools import partial
            # Prog = partial(ub.ProgIter, verbose=3)
            Prog = tqdm
            pbar = Prog(enumerate(dataloader_iter), total=len(eval_dataloader), desc='predict rutgers')
            for batch_index, batch in pbar:
                outputs = batch

                images = outputs['inputs']['im'].data[0]
                original_width, original_height = outputs['tr'].data[0][0]['space_dims']

                images = images.clone()

                bs, c, t, h, w = images.shape

                image1 = images[:, :, 0, :, :]
                image1 = image1.to(self.device)

                input_mask = image1.isnan()

                # Cannot input nan values to the model, so keep them as their
                # imputed value
                image1 = nan_normalize(
                    image1, dim=1, p=2,
                    imputation={'method': 'mean', 'dim': (0, 2, 3)},
                    keepna=False, mask=input_mask)

                output1, outputs_layers = self.model(image1)  # [B,22,150,150]

                # replace model outputs with nans in spatial locations
                output_mask = input_mask.any(dim=1, keepdims=True).expand_as(output1)
                output1[output_mask] = float('nan')

                # print(f"output: {output1.shape}, type: {output1.dtype}")

                bs, c, h, w = output1.shape
                output1_to_save = output1.permute(0, 2, 3, 1).cpu().detach().numpy()
                if self.save_raw_features:
                    # up3_to_save = outputs_layers['up3']
                    up5_to_save = outputs_layers['up5']

                    # up3_to_save = F.interpolate(up3_to_save, size=output1.shape[-2:], mode='bilinear', align_corners=False)
                    up5_to_save = F.interpolate(up5_to_save, size=output1.shape[-2:], mode='bilinear', align_corners=False)

                    # up3_to_save = up3_to_save.cpu().detach().numpy()
                    up5_to_save = up5_to_save.cpu().detach().numpy()
                    # bs, c_up3, h, w  = up3_to_save.shape
                    bs, c_up5, h, w  = up5_to_save.shape

                # print(f"output1_to_save: {output1_to_save.shape}")
                # print(f"output1_to_save min: {output1_to_save.min()}, max: {output1_to_save.max()}")
                # output_show = (output1_to_save - output1_to_save.min()) / (output1_to_save.max() - output1_to_save.min())
                # image_show1 = np.transpose(image1.cpu().detach().numpy()[0, :, :, :], (1, 2, 0))[:, :, :3]
                # image_show1 = (image_show1 - image_show1.min()) / (image_show1.max() - image_show1.min())
                # import matplotlib.pyplot as plt
                # fig = plt.figure()
                # ax1 = fig.add_subplot(1,2,1)
                # ax2 = fig.add_subplot(1,2,2)
                # ax1.imshow(image_show1)
                # ax2.imshow(output_show[0,:,:,:3])
                # plt.show()

                if self.write_probs:
                    for b in range(bs):
                        if len(current_gids) == 0:
                            current_gids = outputs['tr'].data[0][b]['gids']
                        else:
                            previous_gids = current_gids
                            current_gids = outputs['tr'].data[0][b]['gids']
                            mutually_exclusive = (set(previous_gids) - set(current_gids))
                            for gid in mutually_exclusive:
                                pbar.set_postfix_str('finalized {}'.format(len(self.finalized_gids)))
                                seen.add(gid)
                                writer.submit(self.finalize_image, gid)
                                # self.finalize_image(gid)

                        for gid in current_gids:
                            output = output1_to_save[b, :, :, :]

                            if self.save_raw_features:
                                # up3 = up3_to_save[b, :, :, :]
                                up5 = up5_to_save[b, :, :, :]

                                # up3 = np.transpose(up3, (1, 2, 0))
                                up5 = np.transpose(up5, (1, 2, 0))

                            if gid not in self.stitcher_dict.keys():
                                self.stitcher_dict[gid] = kwarray.Stitcher(
                                    (*outputs['tr'].data[0][b]['space_dims'],
                                     self.num_classes),
                                    device='numpy')

                                if self.save_raw_features:
                                    # self.stitcher_dict_up3[gid] = kwarray.Stitcher(
                                    #     (*outputs['tr'].data[0][b]['space_dims'], c_up3),
                                    #     device='numpy')
                                    self.stitcher_dict_up5[gid] = kwarray.Stitcher(
                                        (*outputs['tr'].data[0][b]['space_dims'], c_up5),
                                        device='numpy')
                            slice_ = outputs['tr'].data[0][b]['space_slice']

                            from geowatch.tasks.fusion.predict import CocoStitchingManager
                            CocoStitchingManager._stitcher_center_weighted_add(self.stitcher_dict[gid], slice_, output)

                            if self.save_raw_features:
                                # CocoStitchingManager._stitcher_center_weighted_add(self.stitcher_dict_up3[gid], slice_, up3)
                                CocoStitchingManager._stitcher_center_weighted_add(self.stitcher_dict_up5[gid], slice_, up5)

        writer.wait_until_finished()  # prevents a race condition

        if self.write_probs:
            # Finalize any remaining images
            for gid in Prog(list(self.stitcher_dict.keys()), desc='finish finalization'):
                # self.finalize_image(gid)
                writer.submit(self.finalize_image, gid)
                pbar.set_postfix_str('finalized {}'.format(len(self.finalized_gids)))

        writer.wait_until_finished()

        if self.cache:
            # Still need to update the kwcoco file for all of the hit gids
            for gid in Prog(hit_gids, desc='update coco file for cached images'):
                self.finalize_image(gid, cached=True)

        # export predictions to a new kwcoco file
        self.output_coco_dataset._invalidate_hashid()
        self.output_coco_dataset.dump(self.output_coco_dataset.fpath, newlines=True)

        return

    def forward(self) -> tuple:
        """forward pass for all epochs

        Args:
            cometml_experiment (object): comet ml experiment for logging
            world_size (int, optional): for distributed training. Defaults to 8.

        Returns:
            tuple: (train losses, validation losses, mIoU)
        """

        if self.config['procedures']['validate']:
            self.eval()
        return


def impute(a, imputation='zero', mask=None):
    """
    Replaces nan values according to a imputation method

    Args:
        a (Tensor): input data

        imputation (dict | str):
            dictionary containing keys:
                method (str): either zeros or mean

            if this is a string, it becomes the method in an imputation
            dictionary created with auto defaults.

        mask (Tensor):
            precomputed nan mask
    """
    if mask is None:
        mask = torch.isnan(a)

    if isinstance(imputation, str):
        imputation = {
            'method': imputation
        }
    imputation_method = imputation['method']
    if imputation_method == 'zero':
        out = torch.nan_to_num(a)
    elif imputation_method == 'mean':
        out = a.clone()
        mean_dims = imputation.get('dim', None)
        if mean_dims is None:
            mean = a.nanmean()
            out[mask] = mean
        else:
            fill_values = a.nanmean(dim=mean_dims, keepdims=True).expand_as(out)
            out[mask] = fill_values[mask]
    else:
        raise KeyError(imputation_method)
    return out


def nan_normalize(a, dim, p=2, imputation='zero', assume_nans=False,
                  keepna=True, mask=None):
    """
    Like torch.functional.normalize, but handles nans

    Args:
        a (Tensor): input data

        dim (int): dimension to normalize over

        p (int): type of norm

        imputation (dict | str):

            See :func:`impute`

            dictionary containing keys:
                method (str): either zeros or mean

            if this is a string, it becomes the method in an imputation
            dictionary created with auto defaults.

        assume_nans (bool):
            If true, skips the check if any nans exist and assume they do.
            Otherwise we check if there are no nans and just use normal
            normalize.

        keepna (bool):
            if False, keep the imputed results rather than re-masking them.

        mask (Tensor):
            if specified, use these as masked values

    Returns:
        Tensor: normalized array

    Example:
        >>> shape = (7, 5, 3)
        >>> a = data = torch.from_numpy(np.arange(np.prod(shape)).reshape(*shape)).float()
        >>> a[0:2, 0:2, 0:2] = float('nan')
        >>> dim = 2
        >>> p = 2
        >>> r1 = nan_normalize(a, dim, p, imputation='zero')
        >>> r2 = nan_normalize(a, dim, p, imputation='mean')
        >>> assert r1.isnan().sum() == a.isnan().sum()
        >>> assert r2.isnan().sum() == a.isnan().sum()

        >>> nan_data = torch.full((3, 2), fill_value=float('nan'))
        >>> dim = 1
        >>> nan_result = nan_normalize(nan_data, dim, p, imputation='mean')
        >>> assert torch.isnan(nan_result).all()

        >>> # Ensure this works when no nans exist
        >>> clean_data = torch.rand(3, 2)
        >>> v1 = nan_normalize(clean_data, dim, p, imputation='mean')
        >>> v2 = nan_normalize(clean_data, dim, p, imputation='mean', assume_nans=True)
    """
    if mask is None:
        mask = torch.isnan(a)
    if assume_nans or mask.any():
        if isinstance(imputation, str):
            imputation = {
                'method': imputation
            }
        # mean_dims = imputation.get('dim', 'auto')
        # if mean_dims == 'auto':
        #     mean_dims = tuple([i for i in range(len(a.shape)) if i != dim])
        out = impute(a, imputation=imputation, mask=mask)
        F.normalize(out, dim=dim, p=p, out=out)
        if keepna:
            out[mask] = float('nan')
    else:
        out = F.normalize(a, dim=dim, p=p)
    return out


def make_predict_config(cmdline=False, **kwargs):
    """
    Configuration for material prediction
    """
    from geowatch.utils import configargparse_ext
    from scriptconfig.smartcast import smartcast
    parser = configargparse_ext.ArgumentParser(
        add_config_file_help=False,
        description='Prediction script for the fusion task',
        auto_env_var_prefix='WATCH_RUTGERS_MATERIAL_PREDICT_',
        add_env_var_help=True,
        formatter_class='raw',
        config_file_parser_class='yaml',
        args_for_setting_config_path=['--config'],
        args_for_writing_out_config_file=['--dump'],
    )
    parser.add_argument("--test_dataset", default=None, help='path of the dataset we are going to run inference on')
    parser.add_argument("--pred_dataset", default=None, help='path of the dataset that we are going to write with predictions')
    parser.add_argument("--default_config_key", default=None, help='can be main or iarpa')
    parser.add_argument("--feat_dpath", type=str, help='path to dump asset files. If unspecified, choose a path adjacent to pred_dataset')
    # parser.add_argument("--tag", default='change_prob')
    # parser.add_argument("--package_fpath", type=pathlib.Path)
    parser.add_argument("--export_raw_features", default=smartcast, type=int, help='exporting raw features before classification head')

    # TODO: use torch packages instead
    parser.add_argument("--checkpoint_fpath", type=str, help='path to checkpoint file')
    parser.add_argument("--devices", default=None, help="lightning devices")

    parser.add_argument("--batch_size", default=1, type=int, help="prediction batch size")
    parser.add_argument("--num_workers", default=0, type=str, help="data loader workers, can be set to auto")

    parser.add_argument("--compress", default='DEFLATE', type=str, help="type of gdal compression to use")
    parser.add_argument("--blocksize", default=128, type=str, help="gdal COG blocksize")

    parser.add_argument("--cache", default=False, type=smartcast, help="if True enables caching of predictions in case of a crash")
    # parser.add_argument("--thresh", type=float, default=0.01)

    parser.set_defaults(**kwargs)
    default_args = None if cmdline else []
    args, _ = parser.parse_known_args(default_args)

    assert args.test_dataset is not None, 'must specify path to dataset to predict on'
    assert args.pred_dataset is not None, 'must specify path to dataset to predict on'
    assert args.checkpoint_fpath is not None, 'must specify the path to the checkpoint'

    return args


def hardcoded_default_configs(default_config_key):
    # HACK: THIS IS NOT ROBUST
    from geowatch.tasks import rutgers_material_seg
    from os.path import dirname, join
    module_dpath = dirname(rutgers_material_seg.__file__)
    main_config_path = join(module_dpath, "./configs/main.yaml")
    print('main_config_path = {!r}'.format(main_config_path))
    initial_config = utils.load_yaml_as_dict(main_config_path)
    experiment_config_path = join(module_dpath, f"./configs/{default_config_key}.yaml")
    experiment_config = utils.config_parser(experiment_config_path, experiment_type="training")
    config = {**initial_config, **experiment_config}
    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    return config


def build_evaler(cmdline=False, **kwargs):
    """
    """
    args = make_predict_config(cmdline=cmdline, **kwargs)
    print('args.__dict__ = {}'.format(ub.urepr(args.__dict__, nl=1)))
    config = hardcoded_default_configs(args.default_config_key)

    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    # Hacks to modify the config
    config['training']['pretrained'] = False
    print('config = {}'.format(ub.urepr(config, nl=2)))
    if 0:
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        torch.set_default_dtype(torch.float32)

    from geowatch.utils.lightning_ext import util_device
    from kwutil import util_resources
    devices = util_device.coerce_devices(args.devices)
    num_workers = util_parallel.coerce_num_workers(args.num_workers)
    if len(devices) > 1:
        print('args.devices = {!r}'.format(args.devices))
        print('devices = {!r}'.format(devices))
        raise NotImplementedError('TODO: handle multiple devices')
    device = devices[0]
    if num_workers > 0:
        util_resources.request_nofile_limits()
    print('device = {!r}'.format(device))
    print('num_workers = {!r}'.format(num_workers))

    # Load input kwcoco dataset and prepare the sampler
    input_coco_dset: kwcoco.CocoDataset = kwcoco.CocoDataset.coerce(args.test_dataset)

    # HACK: Filter to remove WV images
    invalid_sensors = {'WV'}
    orig_images = input_coco_dset.images()
    flags = [s not in invalid_sensors for s in orig_images.lookup('sensor_coarse')]
    valid_images = orig_images.compress(flags)
    input_coco_dset = input_coco_dset.subset(valid_images)

    channels = config['data']['channels']
    num_channels = len(channels.split('|'))
    config['training']['num_channels'] = num_channels

    # HACK!!!!
    # THIS IS WHY WE SAVE METADATA WITH THE MODEL!
    # WE DONT WANT TO HAVE TO FUDGE RECONSTRUCTION IN PRODUCTION!!!
    args.checkpoint_fpath = os.fspath(args.checkpoint_fpath)

    base_path = '/'.join(str(args.checkpoint_fpath).split('/')[:-1])
    pretrain_config_path = f"{base_path}/config.yaml"
    if os.path.isfile(pretrain_config_path):
        pretrain_config = utils.load_yaml_as_dict(pretrain_config_path)
        config['data']['channels'] = pretrain_config['data']['channels']
        # config['training']['model_feats_channels'] = pretrain_config_path['training']['model_feats_channels']

    # Load the model
    print('build model')
    model = build_model(model_name=config['training']['model_name'],
                        backbone=config['training']['backbone'],
                        pretrained=config['training']['pretrained'],
                        num_classes=config['data']['num_classes'],
                        num_groups=config['training']['gn_n_groups'],
                        weight_std=config['training']['weight_std'],
                        beta=config['training']['beta'],
                        num_channels=config['training']['num_channels'],
                        out_dim=config['training']['out_features_dim'],
                        feats=config['training']['model_feats_channels'])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    import netharn as nh
    mounted_model_cls = nh.device.DataSerial
    # mounted_model_cls = nn.DataParallel
    print("model has {} trainable parameters".format(num_params))
    model = mounted_model_cls(model)

    output_coco_fpath = pathlib.Path(args.pred_dataset)

    if args.feat_dpath is None:
        output_feat_dpath = output_coco_fpath.parent / '_assets/rutgers_material_seg'
    else:
        output_feat_dpath = pathlib.Path(args.feat_dpath)

    output_feat_dpath.mkdir(exist_ok=1, parents=True)
    output_coco_fpath.parent.mkdir(exist_ok=1, parents=True)

    print('init output dataset')
    # Create the results dataset as a copy of the test CocoDataset
    output_coco_dataset: kwcoco.CocoDataset = input_coco_dset.copy()
    print('clear output annots')
    # Remove all annotations in the results copy
    output_coco_dataset.clear_annotations()
    print('reroot output dset')
    # Change all paths to be absolute paths
    output_coco_dataset.reroot(absolute=True, check=False)
    output_coco_dataset.fpath = os.fspath(output_coco_fpath)

    imwrite_kw = {
        'compress': args.compress,
        'blocksize': args.blocksize,
    }

    print('make evaluator')
    evaler = Evaluator(
        model,
        checkpoint_fpath=args.checkpoint_fpath,
        input_coco_dset=input_coco_dset,
        output_coco_dataset=output_coco_dataset,
        config=config,
        device=device,
        num_workers=num_workers,
        output_feat_dpath=output_feat_dpath,
        imwrite_kw=imwrite_kw,
        batch_size=args.batch_size,
        cache=args.cache,
        save_raw_features=args.export_raw_features
        )
    return evaler


def main(cmdline=False, **kwargs):
    evaler = build_evaler(cmdline=cmdline, **kwargs)
    evaler.forward()


if __name__ == "__main__":
    main(cmdline=True)
