"""
TODO:
    - [ ] make a nicer DVC registry API and CLI
    - [ ] rename

SeeAlso:
    ../cli/find_dvc.py
    python -m geowatch find_dvc list --hardware=ssd --tags=phase2_data
    python -m geowatch find_dvc list --hardware=hdd --tags=phase2_data
    python -m geowatch find_dvc list --hardware=auto --tags=phase2_data
"""
import ubelt as ub
import warnings
import os


class DataRegistry:
    """
    Provide a quick way of storing and querying for machine specific paths

    Ignore:
        from geowatch.utils.util_data import *  # NOQA
        self = DataRegistry()
        self.read()

        test_dpath = ub.Path.appdir('geowatch/tests/dvc_registry').ensuredir()

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
        print(ub.urepr(self.read()))

        self.query(tags='expt_phase1')
        self.query(tags='expt_phase1', max_results=1)
        self.query()

    """

    def __init__(self, registry_fpath=None):
        if registry_fpath is None:
            old_watch_config_dpath = ub.Path.appdir(type='config', appname='watch')
            new_watch_config_dpath = ub.Path.appdir(type='config', appname='geowatch')
            if old_watch_config_dpath.exists():
                watch_config_dpath = old_watch_config_dpath
                import warnings
                warnings.warn(f'Using old watch config directory {old_watch_config_dpath}. Please move all contents to the new directory {new_watch_config_dpath}')
            else:
                watch_config_dpath = new_watch_config_dpath
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

        path = ub.Path(path).absolute()

        if 'hardware' in kwargs:
            if kwargs['hardware'] == 'auto':
                from geowatch.utils import util_hardware
                info = util_hardware.disk_info_of_path(path)
                if 'hwtype' in info:
                    kwargs['hardware'] = info['hwtype']
                else:
                    print('unable to automatically determine hardware type')
                    kwargs.pop('hardware')

        row = ub.udict({'name': name, 'path': os.fspath(path)}) | self._default_attributes
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
        # Hard coded fallback candidate DVC paths, we will remove this in the
        # future.
        hardcoded_paths = [
            {'name': 'hack_data_hdd', 'hardware': 'hdd', 'tags': 'phase2_data', 'path': ub.Path('~/data/dvc-repos/smart_data_dvc').expand()},
            {'name': 'hack_expt_hdd', 'hardware': 'hdd', 'tags': 'phase2_expt', 'path': ub.Path('~/data/dvc-repos/smart_expt_dvc').expand()},
            {'name': 'hack_data_ssd', 'hardware': 'ssd', 'tags': 'phase2_data', 'path': ub.Path('~/data/dvc-repos/smart_data_dvc-ssd').expand()},
            {'name': 'hack_expt_ssd', 'hardware': 'ssd', 'tags': 'phase2_expt', 'path': ub.Path('~/data/dvc-repos/smart_expt_dvc-ssd').expand()},

            {'name': 'hack_data2', 'tags': 'phase2_data', 'path': ub.Path('~/data/smart_data_dvc/').expand()},
            {'name': 'hack_expt2', 'tags': 'phase2_expt', 'path': ub.Path('~/data/smart_expt_dvc/').expand()},
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
                    ub.urepr(list(unexepcted.keys()), nl=0),
                    ub.urepr(list(self._expected_attrs.keys()), nl=0),
                ))
        query = ub.udict({k: v for k, v in kwargs.items() if v is not None})

        ENABLE_EXPERIMENTAL_SPECIAL_QUERY_LOGIC = 1
        if ENABLE_EXPERIMENTAL_SPECIAL_QUERY_LOGIC:
            special_query = {}
            if query.get('hardware', None) == 'auto':
                special_query['hardware'] = query.pop('hardware')

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

        if ENABLE_EXPERIMENTAL_SPECIAL_QUERY_LOGIC:

            if special_query.get('hardware') == 'auto':
                # Make SSDs have higher priority than everything else
                hardware_to_results = ub.group_items(results, lambda x: x.get('hardware', None))
                hardware_to_max_priority = ub.udict()
                for hardware, subs in hardware_to_results.items():
                    hardware_to_max_priority[hardware] = max([s.get('priority', 0) or 0 for s in subs])
                non_ssd_priority = max(1, 1, *(hardware_to_max_priority - {'ssd'}).values())
                min_ssd_priority = min(0, 0, *(hardware_to_max_priority & {'ssd'}).values())

                for row in hardware_to_results.get('ssd', []):
                    row['priority'] = (row.get('priority', 0) or 0) - min_ssd_priority + non_ssd_priority * 2
                # print('hardware_to_results = {}'.format(ub.urepr(hardware_to_results, nl=2)))

        HACK_JONS_REMOTE_PATTERN = 0
        if HACK_JONS_REMOTE_PATTERN:
            # If we can detect the remote pattern that jon likes (where remote
            # machines are mounted via sshfs in the $HOME/remote/$REMOTENAME
            # directory and the localmachine $HOME is symlinked to via
            # $HOME/remote/$HOSTNAME) then use that version of the paths so its
            # easier to work across multiple machines.
            import platform
            host = platform.node()
            for row in results:
                path = ub.Path(row['path'])
                if path.exists() and f'remote/{host}' not in str(path):
                    remote_base = ub.Path(f'~/remote/{host}').expand()
                    remote_alt = path.shrinkuser(home=remote_base)
                    if remote_alt.exists():
                        row['path'] = os.fspath(remote_alt)

        results = sorted(
            results,
            key=lambda r:
                r['priority']
                if r.get('priority', None) is not None else
                -float('inf'))[::-1]
        return results

    def find(self, on_error="raise", envvar=None, **kwargs):
        name = kwargs.get('name', None)
        if envvar is not None:
            environ_dvc_dpath = os.environ.get(envvar, "")
        else:
            environ_dvc_dpath = None
        if environ_dvc_dpath and name is None:
            results = [ub.Path(environ_dvc_dpath)]
        else:
            results = [ub.Path(r['path']) for r in self.query(**kwargs)]
        if not results:
            print('Error in DataRegistry.find. Listing existing data...')
            print(self.list())
            print('Error in DataRegistry.find. Listing query results...')
            print(self.list(**kwargs))
            print('... for query kwargs = {}'.format(ub.urepr(kwargs, nl=1)))
            raise Exception('No suitable data directory found')

        if kwargs.get('must_exist', True):
            results = [found for found in results if found.exists()]

        if not results:
            if on_error == "raise":
                print('Error in DataRegistry.find. Listing existing data...')
                print(self.list())
                print('Error in DataRegistry.find. Listing query results...')
                print(self.list(**kwargs))
                print('... for query kwargs = {}'.format(ub.urepr(kwargs, nl=1)))
                raise Exception('No existing data directory found')
            else:
                return None
        return results[0]


def find_dvc_dpath(name=ub.NoParam, on_error="raise", **kwargs):
    """
    Return the location of the GEOWATCH DVC Data path if it exists and is in
    a "standard" location.

    NOTE: other team members can add their "standard" locations if they want.

    SeeAlso:
        WATCH_DATA_DPATH=$(geowatch_dvc)

        python -m geowatch.cli.find_dvc --hardware=hdd
        python -m geowatch.cli.find_dvc --hardware=ssd
    """
    registry = DataRegistry()
    if name is not ub.NoParam:
        kwargs['name'] = name
    return registry.find(on_error=on_error, **kwargs)


def find_smart_dvc_dpath(*args, **kw):
    return find_dvc_dpath(*args, **kw)
