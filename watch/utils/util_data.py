"""
TODO:
    - [ ] make a nicer DVC registry API and CLI

SeeAlso:
    ../cli/find_dvc.py
"""
import ubelt as ub
import warnings
import os


class DataRegistry:
    """
    Provide a quick way of storing and querying for machine specific paths

    Ignore:
        from watch.utils.util_data import *  # NOQA
        self = DataRegistry()
        self.read()

        test_dpath = ub.Path.appdir('watch/tests/dvc_registry').ensuredir()

        repo1 = (test_dpath / 'repo1').ensuredir()
        repo2 = (test_dpath / 'repo2').ensuredir()
        repo_hdd2 = (test_dpath / 'repo2-hdd').ensuredir()
        repo_ssd2 = (test_dpath / 'repo2-ssd').ensuredir()
        repo_ffs2 = (test_dpath / 'repo2-ffs').ensuredir()

        self.add('repo1', path=repo1, tags='data_phase1')
        self.add('repo2', path=repo2, tags='expt_phase1')
        self.add('repo_hdd2', path=repo_hdd2, hardware='hdd', tags='expt_phase1')
        self.add('repo_ffs2', path=repo_ffs2, hardware='ffs', tags='expt_phase1', priority=10)
        self.add('repo_ssd2', path=repo_ssd2, hardware='ssd', tags='expt_phase1')
        print(self.pandas())
        print(ub.repr2(self.read()))

        self.query(tags='expt_phase1')
        self.query(tags='expt_phase1', max_results=1)
        self.query()

    """
    def __init__(self, registry_fpath=None):
        if registry_fpath is None:
            watch_config_dpath = ub.Path.appdir(type='config', appname='watch')
            registry_dpath = (watch_config_dpath / 'registry').ensuredir()
            registry_fpath = registry_dpath / 'watch_dvc_registry.shelf'

        self.registry_fpath = registry_fpath
        self._default_attributes = ub.udict({
            'priority': None,
            'hardware': None,
            'tags': None,
        })

    def pandas(self, **kwargs):
        import pandas as pd
        return pd.DataFrame(self.query(**kwargs))

    def _open(self):
        import shelve
        shelf = shelve.open(os.fspath(self.registry_fpath))
        return shelf

    def add(self, name, path, **kwargs):
        if name is None:
            raise ValueError('Must specify a name')
        if path is None:
            raise ValueError('Must specify a path')
        unknown = kwargs - self._default_attributes
        if unknown:
            raise ValueError(f'Unknown kwargs={unknown}')
        row = ub.udict({'name': name, 'path': path}) | self._default_attributes
        row |= (kwargs & row)
        shelf = self._open()
        try:
            shelf[name] = row
        finally:
            shelf.close()

    def set(self, name, path=None, **kwargs):
        """
        Set an attribute of a row
        """
        if name is None:
            raise ValueError('Must specify a name')
        unknown = kwargs - self._default_attributes
        if unknown:
            raise ValueError(f'Unknown kwargs={unknown}')
        row = ub.udict({'name': name, 'path': path}) | self._default_attributes
        row |= (kwargs & row)
        shelf = self._open()
        try:
            existing = shelf[name]
            row |= {existing[k] for k, v in row if v is None}
            shelf[name] = row
        finally:
            shelf.close()

    def remove(self, name):
        """
        Ignore:
            name = 'test'
            path = 'foo/bar'
            hardware = 'fake'
        """
        if name is None:
            raise ValueError('Must specify a name')
        shelf = self._open()
        try:
            shelf.pop(name)
        except Exception as ex:
            warnings.warn('Unable to access shelf: {}'.format(ex))
        finally:
            shelf.close()

    def read(self):
        """
        Ignore:
            name = 'test'
            path = 'foo/bar'
            hardware = 'fake'
        """
        # Hard coded fallback candidate DVC paths
        hardcoded_paths = [
            {'path': ub.Path('/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc'), 'name': 'namek', 'hardware': 'hdd', 'tags': 'phase1'},
            {'path': ub.Path('/media/joncrall/raid/dvc-repos/smart_watch_dvc'), 'name': 'ooo', 'hardware': 'hdd', 'tags': 'phase1'},
            {'path': ub.Path("/media/native/data/data/smart_watch_dvc"), 'name': 'rutgers', 'hardware': None, 'tags': 'phase1'},
            {'path': ub.Path("/localdisk0/SCRATCH/watch/ben/smart_watch_dvc"), 'name': 'uky', 'hardware': None, 'tags': 'phase1'},
            {'path': ub.Path("/data4/datasets/smart_watch_dvc/").expand(), 'name': 'purri', 'hardware': None, 'tags': 'phase1'},

            {'path': ub.Path("$HOME/data/dvc-repos/smart_watch_dvc-ssd").expand(), 'name': 'crall-ssd', 'hardware': 'ssd', 'tags': 'phase1_data'},
            {'path': ub.Path("$HOME/data/dvc-repos/smart_watch_dvc-hdd").expand(), 'name': 'crall-hdd', 'hardware': 'hdd', 'tags': 'phase1_data'},
            {'path': ub.Path("$HOME/data/dvc-repos/smart_watch_dvc").expand(), 'name': 'standard', 'hardware': None, 'tags': 'phase1_data'},

            {'path': ub.Path("$HOME/data/dvc-repos/smart_data_dvc").expand(), 'name': 'drop4_standard', 'hardware': 'hdd', 'tags': 'phase2_data'},
            {'path': ub.Path("$HOME/data/dvc-repos/smart_expt_dvc").expand(), 'name': 'drop4_standard', 'hardware': 'hdd', 'tags': 'phase2_expt'},
        ]

        registry_rows = [row for row in hardcoded_paths if row['path'].exists()]

        shelf = self._open()
        try:
            registry_rows += list(shelf.values())
        except Exception as ex:
            warnings.warn('Unable to access shelf: {}'.format(ex))
        finally:
            shelf.close()

        registry_rows = sorted(
            registry_rows,
            key=lambda r:
                r['priority']
                if r.get('priority', None) is not None else
                -float('inf'))[::-1]
        return registry_rows

    def query(self, **kwargs):
        query = ub.udict({k: v for k, v in kwargs.items() if v is not None})
        results = []
        candidate_rows = self.read()
        for row in candidate_rows:
            if query:
                relevant = ub.udict(row).subdict(query, default=None)
                flag = relevant == query
            else:
                flag = True
            if flag:
                results.append(row)
        return results

    def find(self, on_error="raise", envvar='DVC_DPATH', **kwargs):
        name = kwargs.get('name', None)
        environ_dvc_dpath = os.environ.get(envvar, "")
        if environ_dvc_dpath and name is None:
            results = [ub.Path(environ_dvc_dpath)]
        else:
            results = [ub.Path(r['path']) for r in self.query(**kwargs)]
        if not results:
            raise Exception('dvc_dpath not found')
        existing = [found for found in results if found.exists()]
        if not existing:
            if on_error == "raise":
                raise Exception
            else:
                return None
        return existing[0]


def find_dvc_dpath(name=None, on_error="raise", **kwargs):
    """
    Return the location of the SMART WATCH DVC Data path if it exists and is in
    a "standard" location.

    NOTE: other team members can add their "standard" locations if they want.

    SeeAlso:
        WATCH_DATA_DPATH=$(smartwatch_dvc)

        python -m watch.cli.find_dvc --hardware=hdd
        python -m watch.cli.find_dvc --hardware=ssd
    """
    registry = DataRegistry()
    return registry.find(name=name, on_error=on_error, **kwargs)


find_smart_dvc_dpath = find_dvc_dpath
