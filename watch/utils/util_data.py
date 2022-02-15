import ubelt as ub
import os


def find_smart_dvc_dpath(on_error='raise'):
    """
    Return the location of the SMART WATCH DVC Data path if it exists and is in
    a "standard" location.

    NOTE: other team members can add their "standard" locations if they want.

    SeeAlso:
        python -m watch.cli.find_dvc

        python ~/code/watch/watch/cli/find_dvc.py
    """
    environ_dvc_dpath = os.environ.get('DVC_DPATH', '')
    if environ_dvc_dpath:
        dvc_dpath = ub.Path(environ_dvc_dpath)
    else:
        # Fallback to candidate DVC paths
        candidate_dpaths = [
            ub.Path('$HOME/data/dvc-repos/smart_watch_dvc').expand(),
            ub.Path('/media/native/data/data/smart_watch_dvc'),
        ]
        for cand_dpath in candidate_dpaths:
            if cand_dpath.exists():
                dvc_dpath = cand_dpath

    if not dvc_dpath.exists():
        if on_error == 'raise':
            raise Exception
        else:
            return None
    return dvc_dpath
