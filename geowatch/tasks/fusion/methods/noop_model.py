# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py


def __getattr__(key):
    import watch.tasks.fusion.methods.noop_model as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.tasks.fusion.methods.noop_model as mirror
    return dir(mirror)