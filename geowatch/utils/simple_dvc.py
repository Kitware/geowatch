# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.utils.simple_dvc import SimpleDVC_CLI


def __getattr__(key):
    import watch.utils.simple_dvc as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.utils.simple_dvc as mirror
    return dir(mirror)


if __name__ == '__main__':
    SimpleDVC_CLI.main()
