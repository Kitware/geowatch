# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from geowatch.cli.smartflow.run_bas_datagen import main


def __getattr__(key):
    import geowatch.cli.smartflow.run_bas_datagen as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.cli.smartflow.run_bas_datagen as mirror
    return dir(mirror)


if __name__ == '__main__':
    main()
