# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.cli.pseudolive_consolidate import sys


def __getattr__(key):
    import watch.cli.pseudolive_consolidate as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.cli.pseudolive_consolidate as mirror
    return dir(mirror)


if __name__ == '__main__':
    sys.exit(main())
