# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py


def __getattr__(key):
    import watch.geoannots.geomodels as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.geoannots.geomodels as mirror
    return dir(mirror)