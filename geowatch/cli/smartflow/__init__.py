# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py


def __getattr__(key):
    import watch.cli.smartflow as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.cli.smartflow as mirror
    return dir(mirror)