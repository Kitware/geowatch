# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from geowatch.cli.torch_model_stats import main


def __getattr__(key):
    import geowatch.cli.torch_model_stats as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.cli.torch_model_stats as mirror
    return dir(mirror)


if __name__ == '__main__':
    main(cmdline=True)
