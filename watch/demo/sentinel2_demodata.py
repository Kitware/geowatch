# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py


def __getattr__(key):
    import geowatch.demo.sentinel2_demodata as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.demo.sentinel2_demodata as mirror
    return dir(mirror)