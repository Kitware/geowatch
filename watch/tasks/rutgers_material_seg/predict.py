#!/usr/bin/env python
r"""
Prediction script for Rutgers Material Semenatic Segmentation Models

CommandLine:

    export CUDA_VISIBLE_DEVICES=1

    DVC_DPATH=$(python -m watch.cli.find_dvc)

    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15
    BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json

    RUTGERS_MATERIAL_MODEL_FPATH="$DVC_DPATH/models/rutgers/rutgers_peri_materials_v3/experiments_epoch_18_loss_59.014100193977356_valmF1_0.18694573888313187_valChangeF1_0.0_time_2022-02-01-01:53:20.pth"

    RUTGERS_MATERIAL_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/rutgers_material_seg_v3.kwcoco.json

    # Generate Rutgers Features
    python -m watch.tasks.rutgers_material_seg.predict \
        --test_dataset=$BASE_COCO_FPATH \
        --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
        --default_config_key=iarpa \
        --pred_dataset=$RUTGERS_MATERIAL_COCO_FPATH \
        --num_workers=avail \
        --batch_size=4 --gpus 1

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
import watch.tasks.rutgers_material_seg.utils.utils as utils
from watch.utils import util_parallel
from watch.tasks.rutgers_material_seg.models import build_model
from watch.tasks.rutgers_material_seg.datasets.iarpa_contrastive_dataset import SequenceDataset
import torch.nn.functional as F

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class Evaluator(object):
    def __init__(self,
                 model: object,
                 eval_loader: torch.utils.data.DataLoader,
                 output_coco_dataset: kwcoco.CocoDataset,
                 write_probs : bool = True,
                 device=None,
                 config : dict = None,
                 output_feat_dpath : pathlib.Path = None,
                 imwrite_kw={},
                 num_workers=0):
        """Evaluator class

        Args:
            model (object): trained or untrained model
            eval_loader (torch.utils.data.DataLader): loader with evaluation data
            optimizer (object): optimizer to train with
            scheduler (object): scheduler to train with
        """

        self.model = model
        self.eval_loader = eval_loader
        self.num_workers = num_workers
        self.output_coco_dataset = output_coco_dataset
        self.write_probs = write_probs
        self.device = device
        self.config = config
        self.num_classes = self.config['data']['num_classes']
        self.output_feat_dpath = output_feat_dpath
        self.stitcher_dict = {}
        self.finalized_gids = set()
        self.imwrite_kw = imwrite_kw

        # Hack together a channel code
        self.chan_code = '|'.join(['matseg_{}'.format(i) for i in range(self.num_classes)])

    @profile
    def finalize_image(self, gid):
        self.finalized_gids.add(gid)
        stitcher = self.stitcher_dict[gid]
        recon = stitcher.finalize()
        self.stitcher_dict.pop(gid)

        save_path = self.output_feat_dpath / f'{gid}.tif'

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
            'channels': self.chan_code,
            'warp_aux_to_img': warp_aux_to_img.concise(),
        }
        auxiliary = img.setdefault('auxiliary', [])
        auxiliary.append(aux)

    @profile
    def eval(self) -> tuple:
        """evaluate a single epoch

        Args:

        Returns:
            None
        """
        current_gids = []
        previous_gids = []

        self.model.eval()

        dataloader_iter = iter(self.eval_loader)
        writer = util_parallel.BlockingJobQueue(max_workers=self.num_workers)

        seen = set()

        with torch.no_grad():
            # Prog = ub.ProgIter
            Prog = tqdm
            pbar = Prog(enumerate(dataloader_iter), total=len(self.eval_loader), desc='predict rutgers')
            for batch_index, batch in pbar:
                outputs = batch
                break

                images = outputs['inputs']['im'].data[0]
                original_width, original_height = outputs['tr'].data[0][0]['space_dims']

                images = images.clone()

                bs, c, t, h, w = images.shape

                image1 = images[:, :, 0, :, :]
                image1 = image1.to(self.device)

                if 0:
                    image1[0:2, 0:2, 0:2, 0:2] = np.nan

                input_mask = image1.isnan()
                # image1 = utils.stad_image(image1)

                # NOTE: needs to be modified to handle NaNs
                # image1 = F.normalize(image1, dim=1, p=2)

                # Cannot input nan values to the model, so keep them as their
                # imputed value
                image1 = nan_normalize(
                    image1, dim=1, p=2,
                    imputation={'method': 'mean', 'dim': (0, 2, 3)},
                    keepna=False, mask=input_mask)

                output1 = self.model(image1)  # [B,22,150,150]

                # replace model outputs with nans in spatial locations
                output_mask = input_mask.any(dim=1, keepdims=True).expand_as(output1)
                output1[output_mask] = float('nan')

                # print(f"output: {output1.shape}, type: {output1.dtype}")

                bs, c, h, w = output1.shape
                output1_to_save = output1.permute(0, 2, 3, 1).cpu().detach().numpy()
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
                            if gid not in self.stitcher_dict.keys():
                                self.stitcher_dict[gid] = kwarray.Stitcher(
                                    (*outputs['tr'].data[0][b]['space_dims'],
                                     self.num_classes),
                                    device='numpy')
                            slice_ = outputs['tr'].data[0][b]['space_slice']

                            from watch.utils import util_kwimage
                            weights = util_kwimage.upweight_center_mask(output.shape[0:2])[..., None]
                            self.stitcher_dict[gid].add(slice_, output, weight=weights)

                            # weights = kwimage.gaussian_patch(output.shape[0:2])[..., None]
                            # self.stitcher_dict[gid].add(slice_, output, weight=weights)

        if self.write_probs:
            # Finalize any remaining images
            for gid in Prog(list(self.stitcher_dict.keys()), desc='finish finalization'):
                # self.finalize_image(gid)
                writer.submit(self.finalize_image, gid)
                pbar.set_postfix_str('finalized {}'.format(len(self.finalized_gids)))

        writer.wait_until_finished()

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
    from watch.utils import configargparse_ext
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

    # TODO: use torch packages instead
    parser.add_argument("--checkpoint_fpath", type=str, help='path to checkpoint file')
    parser.add_argument("--gpus", default=None, help="todo: hook up to lightning")

    parser.add_argument("--batch_size", default=1, type=int, help="prediction batch size")
    parser.add_argument("--num_workers", default=0, type=str, help="data loader workers, can be set to auto")

    parser.add_argument("--compress", default='RAW', type=str, help="type of gdal compression to use")
    parser.add_argument("--blocksize", default=64, type=str, help="gdal COG blocksize")
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
    from watch.tasks import rutgers_material_seg
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
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.tasks.rutgers_material_seg.predict import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> checkpoint_fpath = dvc_dpath / 'models/rutgers/rutgers_peri_materials_v3/experiments_epoch_18_loss_59.014100193977356_valmF1_0.18694573888313187_valChangeF1_0.0_time_2022-02-01-01:53:20.pth'
        >>> #checkpoint_fpath = dvc_dpath / 'models/rutgers/experiments_epoch_62_loss_0.09470022770735186_valmIoU_0.5901660531463717_time_2021101T16277.pth'
        >>> src_coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json'
        >>> dst_coco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/mat_test.kwcoco.json'
        >>> cmdline = False
        >>> num_workers = 'avail'
        >>> # num_workers = 0
        >>> kwargs = dict(
        >>>     default_config_key='iarpa',
        >>>     checkpoint_fpath=checkpoint_fpath,
        >>>     test_dataset=src_coco_fpath,
        >>>     pred_dataset=dst_coco_fpath,
        >>> )
        >>> evaler = build_evaler(cmdline=cmdline, **kwargs)
        >>> self = evaler
        >>> evaler.forward()
    """
    args = make_predict_config(cmdline=cmdline, **kwargs)
    print('args.__dict__ = {}'.format(ub.repr2(args.__dict__, nl=1)))
    config = hardcoded_default_configs(args.default_config_key)

    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    # Hacks to modify the config
    config['training']['pretrained'] = False
    print(config)
    if 0:
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        torch.set_default_dtype(torch.float32)

    from watch.utils.lightning_ext import util_device
    from watch.utils.lightning_ext import util_globals
    devices = util_device.coerce_devices(args.gpus)
    num_workers = util_globals.coerce_num_workers(args.num_workers)
    if len(devices) > 1:
        raise NotImplementedError('TODO: handle multiple devices')
    device = devices[0]
    if num_workers > 0:
        util_globals.request_nofile_limits()

    # Load input kwcoco dataset and prepare the sampler
    input_coco_dset = kwcoco.CocoDataset.coerce(args.test_dataset)
    sampler = ndsampler.CocoSampler(input_coco_dset)

    window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size'])  # [t,h,w]
    input_dims = (config['data']['image_size'], config['data']['image_size'])

    channels = config['data']['channels']
    num_channels = len(channels.split('|'))
    config['training']['num_channels'] = num_channels
    dataset = SequenceDataset(sampler, window_dims, input_dims, channels,
                              training=False, window_overlap=0.3,
                              inference_only=True)
    print(dataset.__len__())

    eval_dataloader = dataset.make_loader(
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    # HACK!!!!
    # THIS IS WHY WE SAVE METADATA WITH THE MODEL!
    # WE DONT WANT TO HAVE TO FUDGE RECONSTRUCTION IN PRODUCTION!!!
    args.checkpoint_fpath = os.fspath(args.checkpoint_fpath)
    checkpoint_state = torch.load(args.checkpoint_fpath)

    # num_classes = checkpoint_state['model']['module.outc.conv.weight'].shape[0]
    # out_features_dim = checkpoint_state['model']['module.features_outc.conv.weight'].shape[0]
    # config['data']['num_classes'] = num_classes
    # config['training']['out_features_dim'] = out_features_dim

    base_path = '/'.join(str(args.checkpoint_fpath).split('/')[:-1])
    pretrain_config_path = f"{base_path}/config.yaml"
    if os.path.isfile(pretrain_config_path):
        pretrain_config = utils.load_yaml_as_dict(pretrain_config_path)
        config['data']['channels'] = pretrain_config['data']['channels']
        # config['training']['model_feats_channels'] = pretrain_config_path['training']['model_feats_channels']

    # Load the model
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

    model.load_state_dict(checkpoint_state['model'])
    # print(f"loadded model succeffuly from: {pretrain_config_path}")
    # print(f"Missing keys from loaded model: {missing_keys}, unexpected keys: {unexpexted_keys}")

    print('device = {!r}'.format(device))
    model.to(device)

    output_coco_fpath = pathlib.Path(args.pred_dataset)

    if args.feat_dpath is None:
        output_feat_dpath = output_coco_fpath.parent / '_assets/rutgers_material_seg'
    else:
        output_feat_dpath = pathlib.Path(args.feat_dpath)

    output_feat_dpath.mkdir(exist_ok=1, parents=True)
    output_coco_fpath.parent.mkdir(exist_ok=1, parents=True)

    # Create the results dataset as a copy of the test CocoDataset
    output_coco_dataset = input_coco_dset.copy()
    # Remove all annotations in the results copy
    output_coco_dataset.clear_annotations()
    # Change all paths to be absolute paths
    output_coco_dataset.reroot(absolute=True)
    output_coco_dataset.fpath = os.fspath(output_coco_fpath)

    imwrite_kw = {
        'compress': args.compress,
        'blocksize': args.blocksize,
    }

    evaler = Evaluator(
        model,
        eval_dataloader,
        output_coco_dataset=output_coco_dataset,
        config=config,
        device=device,
        num_workers=num_workers,
        output_feat_dpath=output_feat_dpath,
        imwrite_kw=imwrite_kw,
    )
    return evaler


def main(cmdline=False, **kwargs):
    evaler = build_evaler(cmdline=False, **kwargs)
    evaler.forward()

if __name__ == "__main__":
    main(cmdline=True)
