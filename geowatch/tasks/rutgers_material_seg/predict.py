# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
def __getattr__(key):
    import watch.tasks.rutgers_material_seg.predict as mirror
    return getattr(mirror, key)