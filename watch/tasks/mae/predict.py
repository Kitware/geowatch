# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from geowatch.tasks.mae.predict import main


def __getattr__(key):
    import geowatch.tasks.mae.predict as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.tasks.mae.predict as mirror
    return dir(mirror)


if __name__ == '__main__':
    main()
