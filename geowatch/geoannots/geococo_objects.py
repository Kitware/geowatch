# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
def __getattr__(key):
    import watch.geoannots.geococo_objects as mirror
    return getattr(mirror, key)