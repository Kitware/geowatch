# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.cli.mlops_cli import main


def __getattr__(key):
    import watch.cli.mlops_cli as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.cli.mlops_cli as mirror
    return dir(mirror)


if __name__ == '__main__':
    main()
