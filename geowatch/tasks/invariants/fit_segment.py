from .segmentation_model import segmentation_model
from argparse import Namespace
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from datetime import date


def main(args):
    if isinstance(args, dict):
        args = Namespace(**args)

    log_dir = '{}/{}/{}/{}/{}'.format(
        args.save_dir,
        args.dataset,
        'multi_image_segment',
        'position_' + str(args.positional_encoding),
        str(date.today()),
        )

    logger = TensorBoardLogger(log_dir)

    model = segmentation_model(hparams=args)

    checkpoint_callback = ModelCheckpoint(monitor='val_epoch_f1', mode='max', save_top_k=1, save_last=True)
    lr_logger = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=logger,
                                            callbacks=[checkpoint_callback, lr_logger],
                                            log_every_n_steps=10,
                                            check_val_every_n_epoch=args.check_val_every_n_epoch)
    trainer.fit(model)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    ###train hyperparameters
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--learning_rate', type=float, default=.0001)
    parser.add_argument('--save_dir', default='geowatch/tasks/invariants/logs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--drop_rate', type=float, default=.1)

    ###dataset
    parser.add_argument('--dataset', type=str, help='Choose from: spacenet, onera, or kwcoco.', default='kwcoco')

    ### kwcoco arguments
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--vali_dataset', type=str, default='')
    parser.add_argument('--sensor', type=str, nargs='+', default=['S2', 'L8'])
    parser.add_argument('--bands', type=str, nargs='+', default=['shared'])

    ### spacenet arguments
    parser.add_argument('--remove_clouds', help='spacenet specific argument', action='store_true')
    parser.add_argument('--normalize_spacenet', help='spacenet specific argument', action='store_true')

    ### onera arguments
    parser.add_argument('--onera_data_folder', help='Path to Onera. Only relevant if train_dataset and/or vali_dataset are onera.', type=str, default='/localdisk0/SCRATCH/watch/onera/')

    #To do: allow for pretrained weights in this architecture
    ### pretraining arguments
    #     parser.add_argument('--pretrained_checkpoint', type=str, help='path to pretrained checkpoint. Leave blank for change detection training without pretraining.', default='')
    #     parser.add_argument('--pretrained_multihead', action='store_true', help='indicate if the pretrained checkpoint was trained in a multihead fashion')
    #     parser.add_argument('--pretrained_encoder_only', action='store_true')

    ### main argument
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_channels', type=int, default=6)
    parser.add_argument('--pos_class_weight', type=float, help='Weight on positive class for segmentation. Only used on binary labels.', default=1)
    parser.add_argument('--num_images', type=int, default=2)
    parser.add_argument('--attention_layers', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--positional_encoding', action='store_true')
    parser.add_argument('--positional_encoding_mode', type=str, help='addition or concatenation', default='concatenation')
    parser.add_argument('--binary', help='Condense annotations to binary as opposed to site classification. Choose 0 to use classification labels.', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--dataset_style', type=str, default='gridded')
    parser.add_argument('--ignore_boundary', type=int, default=3)
    parser.add_argument('--bas', type=int, default=1)

    args = parser.parse_args()
    main(args)
