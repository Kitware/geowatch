# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
def __getattr__(key):
    import watch.utils.lightning_ext.util_device as mirror
    return getattr(mirror, key)