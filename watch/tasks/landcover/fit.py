# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from geowatch.tasks.landcover.fit import print, sys


def __getattr__(key):
    import geowatch.tasks.landcover.fit as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.tasks.landcover.fit as mirror
    return dir(mirror)


if __name__ == '__main__':
    print('fit not supported', file=sys.stderr)
    sys.exit(1)
