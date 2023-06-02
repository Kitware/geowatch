# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.tasks.uky_temporal_prediction.fit import parser, parser, parser, parser, parser, parser, parser, parser, parser, parser, parser, parser, parser, parser, parser, parser, parser, parser, parser, parser, main


def __getattr__(key):
    import watch.tasks.uky_temporal_prediction.fit as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.tasks.uky_temporal_prediction.fit as mirror
    return dir(mirror)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=.0003)
    parser.add_argument('--gamma', type=float, default=.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--save_dir', type=str, default='logs')
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument(
        '--backbone',
        help='choose from unet, unet_blur',
        default='unet_blur')

    parser.add_argument(
        '--panchromatic',
        help='set flag for using panchromatic WV imagery',
        action='store_true')
    parser.add_argument(
        '--sensor',
        type=str,
        help='choose from WV, LC, or S2',
        default='S2')

    parser.add_argument(
        '--in_channels',
        help='specify the number of channels corresponding to the sensor type',
        type=int,
        default=3)
    parser.add_argument('--train_video', type=int, default=3)
    parser.add_argument('--val_video', type=int, default=5)
    parser.add_argument(
        '--min_time_step',
        help='enforce minimum distance between image pairs',
        type=int,
        default=1)

    parser.add_argument(
        '--train_dataset',
        type=str,
        default='/u/eag-d1/data/watch/drop0_aligned/data.kwcoco.json')

    parser.add_argument(
        '--val_dataset',
        type=str,
        default='/u/eag-d1/data/watch/drop0_aligned/data.kwcoco.json')

    parser.set_defaults(
        gpus=1,
        terminate_on_nan=True,
        check_val_every_n_epochs=1,
        log_every_n_steps=20,
        flush_logs_every_n_steps=20,
        panchromatic=False
    )

    args = parser.parse_args()
    args.default_save_path = os.path.join(args.save_dir, "logs")

    main(args)
