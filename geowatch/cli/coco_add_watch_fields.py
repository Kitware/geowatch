# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.cli.coco_add_watch_fields import main


def __getattr__(key):
    import watch.cli.coco_add_watch_fields as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.cli.coco_add_watch_fields as mirror
    return dir(mirror)


if __name__ == '__main__':
    main()
