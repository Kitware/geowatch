import os
import torch
import datetime
import random
import kwcoco
import kwimage
import ndsampler
import numpy as np
# import ubelt as ub
from torch import nn
from tqdm import tqdm
import watch.tasks.rutgers_material_seg.utils.utils as utils
from watch.tasks.rutgers_material_seg.models import build_model
from watch.tasks.rutgers_material_seg.datasets.iarpa_contrastive_dataset import SequenceDataset
import kwarray

# if 0:
#     torch.backends.cudnn.enabled = False
#     torch.backends.cudnn.deterministic = True
#     torch.set_printoptions(precision=6, sci_mode=False)
#     np.set_printoptions(precision=3, suppress=True)
# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class Evaluator(object):
    def __init__(self,
                 model: object,
                 eval_loader: torch.utils.data.DataLoader,
                 output_coco_dataset: kwcoco.CocoDataset,
                 write_probs=True,
                 device=None,
                 config=None,
                 output_feat_dpath=None):
        """Evaluator class

        Args:
            model (object): trained or untrained model
            eval_loader (torch.utils.data.DataLader): loader with evaluation data
            optimizer (object): optimizer to train with
            scheduler (object): scheduler to train with
        """

        self.model = model
        self.eval_loader = eval_loader
        self.output_coco_dataset = output_coco_dataset
        self.write_probs = write_probs
        self.device = device
        self.config = config
        self.output_feat_dpath = output_feat_dpath

    def diff(self, li1, li2):
        # return list(set(li1) - set(li2)) + list(set(li2) - set(li1))
        return list(set(li2) - set(li1))

    def eval(self) -> tuple:
        """evaluate a single epoch

        Args:

        Returns:
            None
        """
        stitcher_dict = {}
        current_gids = []
        previous_gids = []
        self.model.eval()

        # Hack together a channel code
        chan_code = '|'.join(['matseg_{}'.format(i) for i in range(self.config['data']['num_classes'])])

        with torch.no_grad():
            pbar = tqdm(enumerate(self.eval_loader), total=len(self.eval_loader))
            for batch_index, batch in pbar:
                outputs = batch
                images, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
                original_width, original_height = outputs['tr'].data[0][0]['space_dims']

                # print(outputs['tr'].data[0])

                mask = torch.stack(mask)
                mask = mask.long().squeeze(1)

                bs, c, t, h, w = images.shape
                image1 = images[:, :, 0, :, :]
                image2 = images[:, :, 1, :, :]
                mask1 = mask[:, 0, :, :]  # NOQA
                mask2 = mask[:, 1, :, :]  # NOQA

                images = images.to(self.device)
                image1 = image1.to(self.device)
                image2 = image2.to(self.device)
                mask = mask.to(self.device)

                image1 = utils.stad_image(image1)
                image2 = utils.stad_image(image2)

                output1, features1 = self.model(image1)  # [B,22,150,150]
                output2, features2 = self.model(image2)

                bs, c, h, w = output1.shape
                output1_to_save = output1.permute(0, 2, 3, 1).cpu().detach().numpy()
                output2_to_save = output2.permute(0, 2, 3, 1).cpu().detach().numpy()

                if self.write_probs:

                    for b in range(bs):
                        if len(current_gids) == 0:
                            current_gids = outputs['tr'].data[0][b]['gids']
                        else:
                            previous_gids = current_gids
                            current_gids = outputs['tr'].data[0][b]['gids']
                            mutually_exclusive = self.diff(current_gids, previous_gids)
                            for gid in mutually_exclusive:
                                recon = stitcher_dict[gid].finalize()
                                stitcher_dict.pop(gid)

                                save_path = self.output_feat_dpath / f'{gid}.tiff'
                                save_path = os.fspath(save_path)
                                kwimage.imwrite(save_path, recon,
                                                backend='gdal', space=None,
                                                compress='DEFLATE')

                                aux_height, aux_width = recon.shape[0:2]
                                img = self.output_coco_dataset.index.imgs[gid]
                                warp_aux_to_img = kwimage.Affine.scale(
                                    (img['width'] / aux_width, img['height'] / aux_height))

                                aux = {
                                    'file_name': save_path,
                                    'height': aux_height,
                                    'width': aux_width,
                                    'channels': chan_code,
                                    'warp_aux_to_img': warp_aux_to_img.concise(),
                                }

                                auxiliary = img.setdefault('auxiliary', [])
                                auxiliary.append(aux)
                                # self.output_coco_dataset._invalidate_hashid()

                        for gid, output in zip(current_gids, [output1_to_save[b, :, :, :], output2_to_save[b, :, :, :]]):

                            if gid not in stitcher_dict.keys():
                                stitcher_dict[gid] = kwarray.Stitcher(
                                    (*outputs['tr'].data[0][b]['space_dims'],
                                     self.config['data']['num_classes']))
                            slice_ = outputs['tr'].data[0][b]['space_slice']
                            stitcher_dict[gid].add(slice_, output)

                # masks1 = F.softmax(output1, dim=1)#.detach()
                # masks2 = F.softmax(output2, dim=1)#.detach()
                # # masks1 = F.softmax(features1, dim=1)
                # # masks2 = F.softmax(features2, dim=1)
                # # masks1 = self.high_confidence_filter(masks1, cutoff_top=self.config['high_confidence_threshold']['val_cutoff'])
                # # masks2 = self.high_confidence_filter(masks2, cutoff_top=self.config['high_confidence_threshold']['val_cutoff'])
                # pred1 = masks1.max(1)[1].cpu().detach()#.numpy()
                # pred2 = masks2.max(1)[1].cpu().detach()#.numpy()
                # change_detection_prediction = (pred1!=pred2).type(torch.uint8)

        # export predictions to a new kwcoco file
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
    parser.add_argument("--pred_dataset", default=None)
    parser.add_argument("--default_config_key", default=None, help='can be main or iarpa')
    parser.add_argument("--pred_dpath", type=str, help='path to dump results')
    # parser.add_argument("--tag", default='change_prob')
    # parser.add_argument("--package_fpath", type=pathlib.Path)
    parser.add_argument("--checkpoint_fpath", type=str, help='path to checkpoint file')
    parser.add_argument("--gpus", default=None, help="todo: hook up to lightning")
    # parser.add_argument("--thresh", type=float, default=0.01)

    parser.set_defaults(**kwargs)
    default_args = None if cmdline else []
    args, _ = parser.parse_known_args(default_args)

    assert args.pred_dataset is not None, 'must specify path to dataset to predict on'
    assert args.checkpoint_fpath is not None, 'must specify the path to the checkpoint'
    assert args.pred_dpath is not None, 'Must specify place to write results'

    return args


def hardcoded_default_configs(default_config_key):
    # HACK: THIS IS NOT ROBUST
    from watch.tasks import rutgers_material_seg
    from os.path import dirname, join
    module_dpath = dirname(rutgers_material_seg.__file__)
    main_config_path = join(module_dpath, "./configs/main.yaml")
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
            pred_dataset=ub.expandpath("$HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/data.kwcoco.json"),
            pred_dpath='./test-pred',
        )
    """
    args = make_predict_config(cmdline=cmdline, **kwargs)
    config = hardcoded_default_configs(args.default_config_key)

    config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    # Hacks to modify the config
    config['training']['pretrained'] = False

    if 0:
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        torch.set_default_dtype(torch.float32)

    device = torch.device('cpu' if args.gpus is None else args.gpus)
    device = 0

    # config['device_ids'] = device_ids
    # config['devices_used'] = gpu_devices
    # coco_fpath = config['data'][config['location']]['coco_json']
    # coco_fpath = ub.expandpath(coco_fpath)

    input_coco_dset = kwcoco.CocoDataset.coerce(args.pred_dataset)
    sampler = ndsampler.CocoSampler(input_coco_dset)

    window_dims = (config['data']['time_steps'], config['data']['image_size'], config['data']['image_size'])  # [t,h,w]
    input_dims = (config['data']['image_size'], config['data']['image_size'])

    channels = config['data']['channels']
    num_channels = len(channels.split('|'))
    config['training']['num_channels'] = num_channels
    dataset = SequenceDataset(sampler, window_dims, input_dims, channels)
    print(dataset.__len__())
    eval_dataloader = dataset.make_loader(batch_size=config['training']['batch_size'])

    # HACK!!!!
    # THIS IS WHY WE SAVE METADATA WITH THE MODEL!
    # WE DONT WANT TO HAVE TO FUDGE RECONSTRUCTION IN PRODUCTION!!!
    checkpoint_state = torch.load(args.checkpoint_fpath)
    num_classes = checkpoint_state['model']['module.outc.conv.weight'].shape[0]
    out_features_dim = checkpoint_state['model']['module.features_outc.conv.weight'].shape[0]
    config['data']['num_classes'] = num_classes
    config['training']['out_features_dim'] = out_features_dim

    model = build_model(model_name=config['training']['model_name'],
                        backbone=config['training']['backbone'],
                        pretrained=config['training']['pretrained'],
                        num_classes=config['data']['num_classes'],
                        num_groups=config['training']['gn_n_groups'],
                        weight_std=config['training']['weight_std'],
                        beta=config['training']['beta'],
                        num_channels=config['training']['num_channels'],
                        out_dim=config['training']['out_features_dim'])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model has {} trainable parameters".format(num_params))
    model = nn.DataParallel(model)

    model.load_state_dict(checkpoint_state['model'])

    model.to(device)

    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.eval()
    #         m.weight.requires_grad = False
    #         m.bias.requires_grad = False

    # optimizer = optim.SGD(model.parameters(),
    #                       lr=config['training']['learning_rate'],
    #                       momentum=config['training']['momentum'],
    #                       weight_decay=config['training']['weight_decay'])

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(eval_dataloader),
    #                                                  eta_min=config['training']['learning_rate'])

    # if not config['training']['resume']:
    #     if os.path.isfile(config['training']['resume']):
    #         checkpoint = torch.load(config['training']['resume'])
    #         model.load_state_dict(checkpoint['model'], strict=False)
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         scheduler.load_state_dict(checkpoint['scheduler'])
    #     else:
    #         print("no checkpoint found at {}".format(config['training']['resume']))
    #         exit()

    import pathlib
    output_bundle_dpath = pathlib.Path(args.pred_dpath)
    output_feat_dpath = output_bundle_dpath / '_assets/rutgers_material_seg'
    output_feat_dpath.mkdir(exist_ok=1, parents=True)
    output_coco_fpath = output_bundle_dpath  / 'pred_rutgers_material_seg.kwcoco.json'

    # Create the results dataset as a copy of the test CocoDataset
    output_coco_dataset = input_coco_dset.copy()
    # Remove all annotations in the results copy
    output_coco_dataset.clear_annotations()
    # Change all paths to be absolute paths
    output_coco_dataset.reroot(absolute=True)
    output_coco_dataset.fpath = output_coco_fpath

    evaler = Evaluator(
        model,
        eval_dataloader,
        # optimizer,
        # scheduler,
        output_coco_dataset=output_coco_dataset,
        config=config,
        device=device,
        output_feat_dpath=output_feat_dpath,
    )
    self = evaler  # NOQA
    evaler.forward()

if __name__ == "__main__":
    main()
