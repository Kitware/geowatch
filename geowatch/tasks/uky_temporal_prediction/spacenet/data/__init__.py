# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
def __getattr__(key):
    import watch.tasks.uky_temporal_prediction.spacenet.data as mirror
    return getattr(mirror, key)