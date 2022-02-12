#!/usr/bin/env python
"""
This is a Template for writing training logic.
"""
# package imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from argparse import ArgumentParser, RawTextHelpFormatter
import os

# local imports
from .pretext_model import pretext


def main(args):
    if args.train_dataset is None:
        raise ValueError('train_dataset must contain path to kwcoco file')

    sensor = args.sensor
    if 'S2' in args.sensor and 'L8' in args.sensor:
        sensor = 'S2_L8'

    args.tasks = sorted(args.tasks)
    dataset_name = os.path.basename(os.path.dirname(args.train_dataset))
    log_dir = '{}/{}/{}/{}'.format(
        args.save_dir,
        '_'.join(args.tasks),
        dataset_name,
        sensor,
        )

    model = pretext(hparams=args)
    model.save_package()

    if args.vali_dataset is None:
        ckpt_monitors = (
            ModelCheckpoint(monitor='val_time_accuracy', mode='max', save_top_k=1),
        )
    else:
        ckpt_monitors = (
            ModelCheckpoint(monitor='val_time_accuracy', mode='max', save_top_k=1, save_last=True),
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1),
        )

    lr_logger = LearningRateMonitor(logging_interval='epoch')

    tb_logger = pl_loggers.TensorBoardLogger(log_dir, name='')

    if args.device == 'cpu':
        trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, callbacks=[*ckpt_monitors, lr_logger])
    else:
        trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, gpus=args.gpus, callbacks=[*ckpt_monitors, lr_logger])
    trainer.fit(model)


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.tasks.invariants.fit --help

        python -m watch.tasks.invariants.fit \
            --train_dataset=path/to/train.kwcoco.json \
            --vali_dataset=path/to/vali.kwcoco.json
    """

    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    from scriptconfig.smartcast import smartcast
    ###model hparams
    parser.add_argument('--num_attention_layers', help='Number of attention layers between down convolutions. There can be up to 4 corresponding to each down convolution of the UNet.', type=int, default=4)
    parser.add_argument('--positional_encoding', help='Use positional encoding of time stamps in model. Note: This option should remain False for all currently implemented pretext tasks since the network can learn to cheat based on date information.', action='store_true')
    parser.add_argument('--positional_encoding_mode', help='addition or concatenation.', type=str, default='concatenation')
    ###dataset hparams
    parser.add_argument('--train_dataset', type=str, help="path/to/train.kwcoco.json", required=True)
    parser.add_argument('--vali_dataset', type=str, help="path/to/vali.kwcoco.json", default=None)
    parser.add_argument('--feature_dim_shared', type=int, default=64)
    ###produced features hparams
    parser.add_argument('--feature_dim_each_task', type=int, default=64)
    parser.add_argument('--tasks', nargs='+', help=f'specify which tasks to choose from ({", ".join(pretext.TASK_NAMES)}, or all.\nEx: --tasks {pretext.TASK_NAMES[0]} {pretext.TASK_NAMES[1]}', default=['all'])
    parser.add_argument('--focal_gamma', type=float, help='Focal parameter in loss function for arrow of time task. 0 corresponds to binary cross entropy loss', default=0)
    parser.add_argument('--aot_penalty_weight', type=float, help='Weight to apply to difference of feature map regularization in arrow of time task. Set to 0 to ignore calculations.', default=0)
    parser.add_argument('--aot_penalty_percentage', type=float, help='Percentage of pixels to apply feature map regularization penalty too. Penalty applies to lowest values among differences of feature maps', default=.8)
    ###sensor hparams
    parser.add_argument('--sensor', type=smartcast, nargs='+', default=['S2', 'L8'])
    parser.add_argument('--bands', type=smartcast, help='Choose bands on which to train. Can specify \'all\' for all bands from given sensor, or \'shared\' to use common bands when using both S2 and L8 sensors', nargs='+', default=['shared'])
    ###learning hparams
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--lr_gamma', type=float, default=.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    ###output
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--pca_projection_path', type=str, default='')
    parser.add_argument('--reduction_dim', type=int, default=6)
    parser.add_argument('--use_gridded_dataset', type=int, default=1)
    ###device
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--gpus', type=int, help='gpu(s) to run on', default=1)

    parser.set_defaults(
        terminate_on_nan=True,
        log_every_n_steps=100,
        progress_bar_refresh_rate=1
        )

    args = parser.parse_args()

    args.bands = list(set(args.bands))

    # check save directory
    default_log_path = os.path.join(os.getcwd(), 'watch/tasks/invariants')
    if args.save_dir is None and os.path.exists(default_log_path):
        args.save_dir = os.path.join(default_log_path, 'logs')
    elif args.save_dir is None:
        args.save_dir = 'invariants_logs'

    torch.autograd.set_detect_anomaly(True)
    main(args)
