# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
from geowatch.cli.animate_visualizations import fire


def __getattr__(key):
    import geowatch.cli.animate_visualizations as mirror
    return getattr(mirror, key)


def __dir__():
    import geowatch.cli.animate_visualizations as mirror
    return dir(mirror)


if __name__ == '__main__':
    import fire
    fire.Fire(animate_visualizations)
