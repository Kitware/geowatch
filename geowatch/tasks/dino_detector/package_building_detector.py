# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
def __getattr__(key):
    import watch.tasks.dino_detector.package_dino_detector as mirror
    return getattr(mirror, key)