# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
def __getattr__(key):
    import watch.cli.dag_cli.run_sc_fusion as mirror
    return getattr(mirror, key)