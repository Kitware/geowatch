# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
def __getattr__(key):
    import watch.cli.special.finish_install as mirror
    return getattr(mirror, key)