# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from geowatch.cli.coco_visualize_videos import main


def __getattr__(key):
    import geowatch.cli.coco_visualize_videos as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.cli.coco_visualize_videos as mirror
    return dir(mirror)


if __name__ == '__main__':
    main(cmdline=True)
