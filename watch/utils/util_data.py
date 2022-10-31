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
        # TODO: just use default and filter NoParams
        self._expected_attrs = {
            'name': ub.NoParam,
            'path': ub.NoParam,
        } | self._default_attributes

    def pandas(self, **kwargs):
        import pandas as pd
        df = pd.DataFrame(self.query(**kwargs))
        if len(df):
            df['exists'] = df['path'].apply(lambda p: ub.Path(p).exists())
        return df

    def list(self, **kwargs):
        from rich import print
        print(self.pandas(**kwargs).to_string())

    def _open(self):
        import shelve
        shelf = shelve.open(os.fspath(self.registry_fpath))
        return shelf

    def add(self, name, path, **kwargs):
        if name is None:
            raise ValueError('Must specify a name')
        if path is None:
            raise ValueError('Must specify a path')
        unknown = ub.udict(kwargs) - self._default_attributes
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
        unknown = ub.udict(kwargs) - self._default_attributes
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
            {'name': 'namek', 'hardware': 'hdd', 'tags': 'phase1', 'path': ub.Path('/media/joncrall/raid/home/joncrall/data/dvc-repos/smart_watch_dvc')},
            {'name': 'ooo', 'hardware': 'hdd', 'tags': 'phase1', 'path': ub.Path('/media/joncrall/raid/dvc-repos/smart_watch_dvc')},
            {'name': 'rutgers', 'hardware': None, 'tags': 'phase1', 'path': ub.Path('/media/native/data/data/smart_watch_dvc')},
            {'name': 'uky', 'hardware': None, 'tags': 'phase1', 'path': ub.Path('/localdisk0/SCRATCH/watch/ben/smart_watch_dvc')},
            {'name': 'purri', 'hardware': None, 'tags': 'phase1', 'path': ub.Path('/data4/datasets/smart_watch_dvc')},
            {'name': 'crall-ssd', 'hardware': 'ssd', 'tags': 'phase1_data', 'path': ub.Path('~/data/dvc-repos/smart_watch_dvc-ssd').expand()},
            {'name': 'crall-hdd', 'hardware': 'hdd', 'tags': 'phase1_data', 'path': ub.Path('~/data/dvc-repos/smart_watch_dvc-hdd').expand()},
            {'name': 'phase1_standard', 'hardware': None, 'tags': 'phase1_data', 'path': ub.Path('~/data/dvc-repos/smart_watch_dvc').expand()},

            {'name': 'drop4_data_hdd', 'hardware': 'hdd', 'tags': 'phase2_data', 'path': ub.Path('~/data/dvc-repos/smart_data_dvc').expand()},
            {'name': 'drop4_expt_hdd', 'hardware': 'hdd', 'tags': 'phase2_expt', 'path': ub.Path('~/data/dvc-repos/smart_expt_dvc').expand()},
            {'name': 'drop4_data_ssd', 'hardware': 'ssd', 'tags': 'phase2_data', 'path': ub.Path('~/data/dvc-repos/smart_data_dvc-ssd').expand()},
            {'name': 'drop4_expt_ssd', 'hardware': 'ssd', 'tags': 'phase2_expt', 'path': ub.Path('~/data/dvc-repos/smart_expt_dvc-ssd').expand()},
        ]

        # registry_rows = [row for row in hardcoded_paths if row['path'].exists()]
        registry_rows = hardcoded_paths.copy()

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

    def query(self, must_exist=False, **kwargs):
        unexepcted = ub.udict(kwargs) - self._expected_attrs
        if unexepcted:
            raise ValueError(
                'Unexpected query keywords: {}. Valid keywords are {}'.format(
                    ub.repr2(list(unexepcted.keys()), nl=0),
                    ub.repr2(list(self._expected_attrs.keys()), nl=0),
                ))
        query = ub.udict({k: v for k, v in kwargs.items() if v is not None})
        results = []
        candidate_rows = self.read()
        for row in candidate_rows:
            if query:
                relevant = ub.udict(row).subdict(query, default=None)
                flag = relevant == query
            else:
                flag = True
            if must_exist:
                if not ub.Path(row['path']).exists():
                    flag = False
            if flag:
                results.append(row)

        HACK_JONS_REMOTE_PATTERN = 1
        if HACK_JONS_REMOTE_PATTERN:
            # If we can detect the remote pattern that jon likes (where remote
            # machines are mounted via sshfs in the $HOME/remote/$REMOTENAME
            # directory and the localmachine $HOME is symlinked to via
            # $HOME/remote/$HOSTNAME) then use that version of the paths so its
            # easier to work across multiple machines.
            for row in results:
                path = ub.Path(row['path'])
                if path.exists():
                    import platform
                    host = platform.node()
                    remote_base = ub.Path(f'~/remote/{host}').expand()
                    remote_alt = path.shrinkuser(home=remote_base)
                    if remote_alt.exists():
                        row['path'] = os.fspath(remote_alt)
        return results

    def find(self, on_error="raise", envvar='DVC_DPATH', **kwargs):
        name = kwargs.get('name', None)
        environ_dvc_dpath = os.environ.get(envvar, "")
        if environ_dvc_dpath and name is None:
            results = [ub.Path(environ_dvc_dpath)]
        else:
            results = [ub.Path(r['path']) for r in self.query(**kwargs)]
        if not results:
            print('Error in DataRegistry.find. Listing existing data...')
            print(self.list())
            print('Error in DataRegistry.find. Listing query results...')
            print(self.list(**kwargs))
            print('... for query kwargs = {}'.format(ub.repr2(kwargs, nl=1)))
            raise Exception('No suitable data directory found')

        if kwargs.get('must_exist', True):
            results = [found for found in results if found.exists()]

        if not results:
            if on_error == "raise":
                print('Error in DataRegistry.find. Listing existing data...')
                print(self.list())
                print('Error in DataRegistry.find. Listing query results...')
                print(self.list(**kwargs))
                print('... for query kwargs = {}'.format(ub.repr2(kwargs, nl=1)))
                raise Exception('No existing data directory found')
            else:
                return None
        return results[0]


def find_dvc_dpath(name=ub.NoParam, on_error="raise", **kwargs):
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
    if name is not ub.NoParam:
        kwargs['name'] = name
    return registry.find(on_error=on_error, **kwargs)


def find_smart_dvc_dpath(*args, **kw):
    return find_dvc_dpath(*args, **kw)
