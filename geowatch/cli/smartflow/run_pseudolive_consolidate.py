# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from watch.cli.smartflow.run_pseudolive_consolidate import main


def __getattr__(key):
    import watch.cli.smartflow.run_pseudolive_consolidate as mirror
    return getattr(mirror, key)


def __dir__():
    import watch.cli.smartflow.run_pseudolive_consolidate as mirror
    return dir(mirror)


if __name__ == '__main__':
    main()
