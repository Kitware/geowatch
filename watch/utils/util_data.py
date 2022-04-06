"""
TODO:
    - [ ] make a nicer DVC registry API and CLI

SeeAlso:
    ../cli/find_dvc.py
"""
import ubelt as ub
import warnings
import os


def _open_dvc_registry_shelf():
    import shelve
    watch_config_dpath = ub.Path.appdir(type='config', appname='watch')
    registry_dpath = (watch_config_dpath / 'registry').ensuredir()
    registry_fpath = registry_dpath / 'watch_dvc_registry.shelf'
    shelf = shelve.open(os.fspath(registry_fpath))
    return shelf


def _dvc_registry_add(name, path, hardware=None):
    """
    Ignore:
        name = 'test'
        path = 'foo/bar'
        hardware = 'fake'
    """
    if name is None:
        raise ValueError('Must specify a name')
    shelf = _open_dvc_registry_shelf()
    try:
        shelf[name] = {
            'name': name,
            'path': path,
            'hardware': hardware,
        }
    finally:
        shelf.close()


def _dvc_registry_remove(name):
    """
    Ignore:
        name = 'test'
        path = 'foo/bar'
        hardware = 'fake'
    """
    if name is None:
        raise ValueError('Must specify a name')
    shelf = _open_dvc_registry_shelf()
    try:
        shelf.pop(name)
    except Exception as ex:
        warnings.warn('Unable to access shelf: {}'.format(ex))
    finally:
        shelf.close()


def _dvc_registry_list():
    """
    Ignore:
        name = 'test'
        path = 'foo/bar'
        hardware = 'fake'
    """
    # Hard coded fallback candidate DVC paths
    hardcoded_paths = [
        {'path': ub.Path('/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc'),  'name': 'namek', 'hardware': 'hdd'},
        {'path': ub.Path('/media/joncrall/raid/dvc-repos/smart_watch_dvc'),  'name': 'ooo', 'hardware': 'hdd'},
        {'path': ub.Path("/media/native/data/data/smart_watch_dvc"),  'name': 'rutgers', 'hardware': None},
        {'path': ub.Path("/localdisk0/SCRATCH/watch/ben/smart_watch_dvc"), 'name': 'uky', 'hardware': None},
        {'path': ub.Path("/data4/datasets/smart_watch_dvc/").expand(), 'name': 'purri', 'hardware': None},
        {'path': ub.Path("$HOME/data/dvc-repos/smart_watch_dvc").expand(), 'name': 'crall-ssd', 'hardware': 'ssd'},
        {'path': ub.Path("$HOME/data/dvc-repos/smart_watch_dvc-hdd").expand(), 'name': 'crall-hdd', 'hardware': 'hdd'},
    ]

    candidate_dpaths = [row for row in hardcoded_paths if row['path'].exists()]

    shelf = _open_dvc_registry_shelf()
    try:
        candidate_dpaths += list(_open_dvc_registry_shelf().values())
    except Exception as ex:
        warnings.warn('Unable to access shelf: {}'.format(ex))
    finally:
        shelf.close()

    return candidate_dpaths


def find_smart_dvc_dpath(hardware=None, name=None, on_error="raise"):
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
    if environ_dvc_dpath and name is None:
        dvc_dpath = ub.Path(environ_dvc_dpath)
    else:
        dvc_dpath = None
        candidate_dpaths = _dvc_registry_list()
        for row in candidate_dpaths:
            dpath = ub.Path(row['path'])
            if dpath.exists():
                if hardware is None or hardware == row.get('hardware', None):
                    if name is None or name == row.get('name', None):
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
