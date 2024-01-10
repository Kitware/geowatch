#r -*- coding: utf-8 -*-
r"""
Prediction script for Rutgers Material Semenatic Segmentation Models

CommandLine:

    DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
    KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}
    BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
    RUTGERS_MATERIAL_MODEL_FPATH="$DVC_DPATH/models/rutgers/experiments_epoch_30_loss_0.05691597167379317_valmIoU_0.5694727912477856_time_2021-08-07-09:01:01.pth"
    RUTGERS_MATERIAL_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/rutgers_material_seg.kwcoco.json

    DVC_DPATH=${DVC_DPATH:-/media/native/data/data/smart_watch_dvc}
    KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}
    BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
    RUTGERS_MATERIAL_MODEL_FPATH="/home/native/projects/data/smart_watch/models/experiments_onera/tasks_experiments_onera_2021-10-20-17:15/experiments_epoch_37_loss_7.454268312454223_valmF1_0.7629152048972937_valChangeF1_0.5579948695099214_time_2021-10-20-18:04:59.pth"
    RUTGERS_MATERIAL_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/rutgers_material_seg.kwcoco.json


    # Generate Rutgers Features
    python -m geowatch.tasks.rutgers_material_seg.predict_test \
        --test_dataset=$BASE_COCO_FPATH \
        --checkpoint_fpath=$RUTGERS_MATERIAL_MODEL_FPATH  \
        --default_config_key=iarpa \
        --pred_dataset=$RUTGERS_MATERIAL_COCO_FPATH \
        --num_workers=8 \
        --batch_size=32 --devices auto:1

"""
import os
import torch
import datetime
import random
import kwcoco
import ndsampler
import numpy as np
import ubelt as ub
import pathlib
import geowatch.tasks.rutgers_material_seg.utils.utils as utils
from geowatch.tasks.rutgers_material_seg.models import build_model
from geowatch.tasks.rutgers_material_seg.datasets.iarpa_contrastive_dataset import SequenceDataset


class Evaluator(object):
    def __init__(self,
                 model: object,
                 dataset: object,
                 eval_loader: torch.utils.data.DataLoader,
                 output_coco_dataset: kwcoco.CocoDataset,
                 write_probs : bool = True,
                 device=None,
                 config : dict = None,
                 output_feat_dpath : pathlib.Path = None):
        """Evaluator class

        Args:
            model (object): trained or untrained model
            eval_loader (torch.utils.data.DataLader): loader with evaluation data
            optimizer (object): optimizer to train with
            scheduler (object): scheduler to train with
        """

        self.model = model
        self.dataset = dataset
        self.eval_loader = eval_loader
        self.output_coco_dataset = output_coco_dataset
        self.write_probs = write_probs
        self.device = device
        self.config = config
        self.num_classes = self.config['data']['num_classes']
        self.output_feat_dpath = output_feat_dpath
        self.stitcher_dict = {}
        self.finalized_gids = set()
        self.sensors = ['S2', 'L8']

        # Hack together a channel code
        self.chan_code = '|'.join(['matseg_{}'.format(i) for i in range(self.num_classes)])

    def eval(self) -> tuple:
        """evaluate a single epoch

        Args:

        Returns:
            None
        """
        # current_gids = []
        # previous_gids = []

        self.model.eval()
        image_ids = list(self.dataset.index.imgs.keys())
        valid_images = self.dataset.images(image_ids)
        sensor = 'S2'
        flags = [sensor == _ for _ in valid_images.lookup('sensor_coarse')]
        image_ids = valid_images.compress(flags)
        with torch.no_grad():
            # Prog = ub.ProgIter
            # Prog = tqdm
            # pbar = Prog(enumerate(self.eval_loader), total=len(self.eval_loader), desc='predict rutgers')
            # for batch_index, batch in pbar:
            for gid in image_ids:
                # outputs = batch
                # images, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0][0]
                # original_width, original_height = outputs['tr'].data[0][0]['space_dims']

                # img = self.dataset.index.imgs[gid]
                delayed_image = self.dataset.delayed_load(gid)
                im = delayed_image.finalize()
                image = torch.from_numpy(im.astype('float32')).to(self.device)
                print(f"images: {image.shape}")
                image = image.permute(2, 0, 1)
                image = image.unsqueeze(0)[:, :6, :, :]
                inference_all_crops_params = [tuple([i, j, self.config['evaluation']['inference_window'], self.config['evaluation']['inference_window']]) for i in range(0, image.shape[2]) for j in range(0, image.shape[1])]
                print(f"images: {image.shape}")
                # print(f"mask: {mask.shape}")

                # mask = torch.stack(mask)
                # mask = mask.long().squeeze(1)

                # bs, c, t, h, w = images.shape

                # image1 = images[:, :, 0, :, :]
                # mask1 = mask[:, 0, :, :]  # NOQA

                # image1 = image1.to(self.device)
                # mask = mask.to(self.device)

                # # image1 = utils.stad_image(image1)
                # # image2 = utils.stad_image(image2)

                output1 = self.model(image, image, inference_all_crops_params)
                # import matplotlib.pyplot as plt
                print(f"output: {output1.shape}")
                # output_show = output1[0,1,:,:].cpu().detach().numpy()
                # plt.imshow(output_show)
                # plt.show()

                # bs, c, h, w = output1.shape
                # output1_to_save = output1.permute(0, 2, 3, 1).cpu().detach().numpy()

        # # export predictions to a new kwcoco file
        # self.output_coco_dataset._invalidate_hashid()
        # self.output_coco_dataset.dump(self.output_coco_dataset.fpath, newlines=True)

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


def make_predict_config(cmdline=False, **kwargs):
    """
    Configuration for material prediction
    """
    from geowatch.utils import configargparse_ext
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
    parser.add_argument("--devices", default=None, help="lightning devices")

    parser.add_argument("--batch_size", default=1, type=int, help="prediction batch size")
    parser.add_argument("--num_workers", default=0, type=str, help="data loader workers, can be set to auto")
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


def main(cmdline=True, **kwargs):
    """
    Ignore:
        # Hack in overrides because none of this is parameterized
        # state_dict = torch.load(checkpoint_fpath)
        checkpoint_fpath = ub.expandpath("$HOME/data/dvc-repos/smart_watch_dvc/models/rutgers/experiments_epoch_30_loss_0.05691597167379317_valmIoU_0.5694727912477856_time_2021-08-07-09:01:01.pth")
        cmdline = False
        kwargs = dict(
            default_config_key='iarpa',
            checkpoint_fpath=checkpoint_fpath,
            test_dataset=ub.expandpath("$HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/data.kwcoco.json"),
            pred_dataset='./test-pred/pred.kwcoco.json',
        )
    """
    args = make_predict_config(cmdline=cmdline, **kwargs)
    print('args.__dict__ = {}'.format(ub.urepr(args.__dict__, nl=1)))
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

    from geowatch.utils.lightning_ext import util_device
    devices = util_device.coerce_devices(args.devices)
    if len(devices) > 1:
        raise NotImplementedError('TODO: handle multiple devices')
    device = devices[0]

    input_coco_dset = kwcoco.CocoDataset.coerce(args.test_dataset)
    sampler = ndsampler.CocoSampler(input_coco_dset)

    window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size'])  # [t,h,w]
    input_dims = (config['data']['image_size'], config['data']['image_size'])

    channels = config['data']['channels']
    num_channels = len(channels.split('|'))
    config['training']['num_channels'] = num_channels
    dataset = SequenceDataset(sampler, window_dims, input_dims, channels,
                              training=False)
    print(dataset.__len__())

    from kwutil import util_resources
    from kwutil import util_parallel
    num_workers = util_parallel.coerce_num_workers(args.num_workers)
    if num_workers > 0:
        util_resources.request_nofile_limits()

    eval_dataloader = dataset.make_loader(
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    # HACK!!!!
    # THIS IS WHY WE SAVE METADATA WITH THE MODEL!
    # WE DONT WANT TO HAVE TO FUDGE RECONSTRUCTION IN PRODUCTION!!!
    checkpoint_state = torch.load(args.checkpoint_fpath)  # NOQA

    # num_classes = checkpoint_state['model']['module.outc.conv.weight'].shape[0]
    # out_features_dim = checkpoint_state['model']['module.features_outc.conv.weight'].shape[0]
    # config['data']['num_classes'] = num_classes
    # config['training']['out_features_dim'] = out_features_dim

    base_path = '/'.join(args.checkpoint_fpath.split('/')[:-1])
    pretrain_config_path = f"{base_path}/config.yaml"
    if os.path.isfile(pretrain_config_path):
        pretrain_config = utils.load_yaml_as_dict(pretrain_config_path)
        config['data']['channels'] = pretrain_config['data']['channels']
        # config['training']['model_feats_channels'] = pretrain_config_path['training']['model_feats_channels']

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

    # model.load_state_dict(checkpoint_state['model'])

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

    evaler = Evaluator(
        model,
        dataset=input_coco_dset,
        eval_loader=eval_dataloader,
        output_coco_dataset=output_coco_dataset,
        config=config,
        device=device,
        output_feat_dpath=output_feat_dpath,
    )
    self = evaler  # NOQA
    evaler.forward()


if __name__ == "__main__":
    main()
