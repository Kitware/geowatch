# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py


def __getattr__(key):
    import watch.tasks.fusion.datamodules.temporal_sampling.sampler as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.tasks.fusion.datamodules.temporal_sampling.sampler as mirror
    return dir(mirror)