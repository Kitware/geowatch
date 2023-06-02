# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.tasks.uky_temporal_prediction.spacenet.data.create_splits import parser, main


def __getattr__(key):
    import watch.tasks.uky_temporal_prediction.spacenet.data.create_splits as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.tasks.uky_temporal_prediction.spacenet.data.create_splits as mirror
    return dir(mirror)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/localdisk0/SCRATCH/watch/SpaceNet/7/train/')
    args = parser.parse_args()

    main(args.data_dir)
