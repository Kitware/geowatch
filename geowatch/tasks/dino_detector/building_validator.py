# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.tasks.dino_detector.building_validator import main


def __getattr__(key):
    import watch.tasks.dino_detector.building_validator as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.tasks.dino_detector.building_validator as mirror
    return dir(mirror)


if __name__ == '__main__':
    main()
