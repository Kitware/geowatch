# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.cli.merge_region_models import main


def __getattr__(key):
    import watch.cli.merge_region_models as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.cli.merge_region_models as mirror
    return dir(mirror)


if __name__ == '__main__':
    main(cmdline=True)
