# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from geowatch.cli.find_dvc import __config__


def __getattr__(key):
    import geowatch.cli.find_dvc as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.cli.find_dvc as mirror
    return dir(mirror)


if __name__ == '__main__':
    __config__.main()
