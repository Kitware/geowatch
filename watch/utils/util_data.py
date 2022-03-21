import ubelt as ub
import os


def _open_dvc_registry_shelf():
    import shelve
    import os
    watch_config_dpath = ub.Path.appdir(type='config', appname='watch')
    registry_dpath = (watch_config_dpath / 'registry').ensuredir()
    registry_fpath = registry_dpath / 'watch_dvc_registry.shelf'
    shelf = shelve.open(os.fspath(registry_fpath))
    return shelf


def register_smart_dvc_dpath(name, path, hardware=None):
    """
    TODO:
        - [ ] Add the ability for the user to register a DVC path
        - [ ] Add the ability for the user to unregister a DVC path

    Ignore:
        name = 'test'
        path = 'foo/bar'
        hardware = 'fake'
    """
    shelf = _open_dvc_registry_shelf()
    try:
        shelf[name] = {
            'name': name,
            'path': path,
            'hardware': hardware,
        }
    finally:
        shelf.close()


def find_smart_dvc_dpath(hardware=None, on_error="raise"):
    """
    Return the location of the SMART WATCH DVC Data path if it exists and is in
    a "standard" location.

    NOTE: other team members can add their "standard" locations if they want.

    SeeAlso:
        WATCH_DATA_DPATH=$(python -m watch.cli.find_dvc)

        python -m watch.cli.find_dvc --hardware=hdd
        python -m watch.cli.find_dvc --hardware=ssd
    """
    environ_dvc_dpath = os.environ.get("DVC_DPATH", "")
    if environ_dvc_dpath:
        dvc_dpath = ub.Path(environ_dvc_dpath)
    else:
        # Fallback to candidate DVC paths
        candidate_dpaths = [
            {'path': ub.Path("/media/native/data/data/smart_watch_dvc"),  'name': 'rutgers', 'hardware': None},
            {'path': ub.Path("/localdisk0/SCRATCH/watch/ben/smart_watch_dvc"), 'name': 'uky', 'hardware': None},
            {'path': ub.Path("/data4/datasets/smart_watch_dvc/").expand(), 'name': 'purri', 'hardware': None},
            {'path': ub.Path("$HOME/data/dvc-repos/smart_watch_dvc").expand(), 'name': 'crall', 'hardware': 'ssd'},
            {'path': ub.Path("$HOME/data/dvc-repos/smart_watch_dvc-hdd").expand(), 'name': 'crall', 'hardware': 'hdd'},
        ]

        if 1:
            candidate_dpaths += _open_dvc_registry_shelf().values()

        dvc_dpath = None
        for row in candidate_dpaths:
            dpath = row['path']
            if dpath.exists():
                if hardware is None or hardware == row.get('hardware', None):
                    dvc_dpath = dpath
                    break
        if dvc_dpath is None:
            raise Exception('dvc_dpath not found')

    if not dvc_dpath.exists():
        if on_error == "raise":
            raise Exception
        else:
            return None
    return dvc_dpath
