# Autogenerated via:
# python ~/code/watch/dev/maintain/mirror_package_geowatch.py
def __getattr__(key):
    import watch.tasks.metrics.viz_sc_results as mirror
    return getattr(mirror, key)