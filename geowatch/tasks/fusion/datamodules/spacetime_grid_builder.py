# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
def __getattr__(key):
    import watch.tasks.fusion.datamodules.spacetime_grid_builder as mirror
    return getattr(mirror, key)