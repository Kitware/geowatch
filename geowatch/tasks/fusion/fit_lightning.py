# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.tasks.fusion.fit_lightning import main


def __getattr__(key):
    import watch.tasks.fusion.fit_lightning as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.tasks.fusion.fit_lightning as mirror
    return dir(mirror)


if __name__ == '__main__':
    main()
