# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
def __getattr__(key):
    import watch.tasks.uky_temporal_prediction as mirror
    return getattr(mirror, key)