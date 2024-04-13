def patch_kwutil_toplevel():
    """
    For kwutil versions <= 0.2.6 which don't expose common methods at the top
    level.
    """
    import kwutil
    kwutil.datetime = kwutil.util_time.datetime
    kwutil.timedelta = kwutil.util_time.timedelta
    kwutil.Yaml = kwutil.util_yaml.Yaml
