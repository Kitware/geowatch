# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py


def __getattr__(key):
    import geowatch.tasks.fusion.methods.heterogeneous as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.tasks.fusion.methods.heterogeneous as mirror
    return dir(mirror)