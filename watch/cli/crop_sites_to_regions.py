# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from geowatch.cli.crop_sites_to_regions import main


def __getattr__(key):
    import geowatch.cli.crop_sites_to_regions as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.cli.crop_sites_to_regions as mirror
    return dir(mirror)


if __name__ == '__main__':
    main(cmdline=True)
