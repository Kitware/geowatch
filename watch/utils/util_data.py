import ubelt as ub
import os


def find_smart_dvc_dpath(on_error='raise'):
    """
    Return the location of the SMART WATCH DVC Data path if it exists and is in
    a "standard" location.

    NOTE: other team members can add their "standard" locations if they want.
    """
    _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
    dvc_dpath = os.environ.get('DVC_DPATH', _default)
    dvc_dpath = ub.Path(dvc_dpath)

    if not dvc_dpath.exists():
        if on_error == 'raise':
            raise Exception
        else:
            return None
    return dvc_dpath
