# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
def __getattr__(key):
    import watch.tasks.depth.pl_highres_verify as mirror
    return getattr(mirror, key)