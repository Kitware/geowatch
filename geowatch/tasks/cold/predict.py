# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.tasks.cold.predict import cold_predict_main


def __getattr__(key):
    import watch.tasks.cold.predict as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.tasks.cold.predict as mirror
    return dir(mirror)


if __name__ == '__main__':
    cold_predict_main()
