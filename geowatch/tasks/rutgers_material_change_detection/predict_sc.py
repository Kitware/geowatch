# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.tasks.rutgers_material_change_detection.predict_sc import main


def __getattr__(key):
    import watch.tasks.rutgers_material_change_detection.predict_sc as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.tasks.rutgers_material_change_detection.predict_sc as mirror
    return dir(mirror)


if __name__ == '__main__':
    main()
