# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py


def __getattr__(key):
    import geowatch.gis.spatial_reference as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.gis.spatial_reference as mirror
    return dir(mirror)