"""
Experimental feature where each training run will register itself in a global
registery. We can then provide a command to make it easy to check which
training runs were started on this machine and where they are located.


Usage
-----

To See

.. code:: bash

        python -m geowatch.tasks.fusion.fit_registery peek

"""
import ubelt as ub
import warnings
import scriptconfig as scfg


class Registery:
    """
    Attributes:
        fpath : path to the registery's shelf database.
    """
    __registery_fname__ = 'abstract.shelf'
    __appname__ = 'geowatch'
    __default__ = {}

    def __init__(self, fpath=None):
        if fpath is None:
            app_dpath = ub.Path.appdir(type='config', appname=self.__appname__)
            registry_dpath = (app_dpath / 'registry').ensuredir()
            fpath = registry_dpath / self.__registery_fname__

        self.fpath = fpath
        self._required_attributes = {k for k, v in self.__default__.items() if v is ub.NoParam}

    def pandas(self, **kwargs):
        import pandas as pd
        df = pd.DataFrame(self.query(**kwargs))
        return df

    def list(self, **kwargs):
        from rich import print
        print(self.pandas(**kwargs).to_string())

    def _open(self):
        import os
        import shelve
        shelf = shelve.open(os.fspath(self.fpath))
        return shelf

    def query(self, **kwargs):
        unexepcted = ub.udict(kwargs) - self.__default__
        if unexepcted:
            raise ValueError(
                'Unexpected query keywords: {}. Valid keywords are {}'.format(
                    ub.urepr(list(unexepcted.keys()), nl=0),
                    ub.urepr(list(self.__default__.keys()), nl=0),
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
            if flag:
                results.append(row)

        results = sorted(
            results,
            key=lambda r:
                r['priority']
                if r.get('priority', None) is not None else
                -float('inf'))[::-1]
        return results

    def _items(self):
        items = []
        with self._open() as shelf:
            try:
                items += list(shelf.items())
            except Exception as ex:
                warnings.warn('Unable to access shelf: {}'.format(ex))
        return items


class DictRegistery(Registery):

    def add(self, key, **kwargs):
        unknown = ub.udict(kwargs) - self.__default__
        if unknown:
            raise ValueError(f'Unknown kwargs={unknown}')

        for key in self._required_attributes:
            if key not in kwargs:
                raise ValueError(f'Must specify a {key}')

        row = ub.udict(self.__default__)
        row |= (kwargs & row)
        shelf = self._open()
        try:
            shelf[key] = row
        finally:
            shelf.close()

    def set(self, key, **kwargs):
        """
        Set an attribute of a row
        """
        if key is None:
            raise ValueError('Must specify a key')
        unknown = ub.udict(kwargs) - self.__default__
        if unknown:
            raise ValueError(f'Unknown kwargs={unknown}')
        row = ub.udict({'key': key}) | self.__default__
        row |= (kwargs & row)
        shelf = self._open()
        try:
            existing = shelf[key]
            row |= {existing[k] for k, v in row if v is None}
            shelf[key] = row
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
        registry_rows = []
        with self._open() as shelf:
            try:
                registry_rows += list(shelf.values())
            except Exception as ex:
                warnings.warn('Unable to access shelf: {}'.format(ex))
        return registry_rows


class ListRegistery(Registery):
    """
    Example:
        >>> # A minimal example that inherits from ListRegistery
        >>> from geowatch.tasks.fusion.fit_registery import *  # NOQA
        >>> class DemoListRegistery(ListRegistery):
        >>>     __registery_fname__ = 'geowatch_fit_registery.shelf'
        >>>     __appname__ = 'geowatch/demo'
        >>>     __default__ = ub.udict({
        >>>         'path'     : ub.NoParam,
        >>>         'time'     : None,
        >>>         'uuid'     : None,
        >>>         'config'   : None,
        >>>     })
        >>> #
        >>> self = DemoListRegistery()
        >>> self.fpath
        >>> print(self.fpath.exists())
        >>> self.append(path='foo', time='bar', uuid='blag')
        >>> self.append(path='foo', time='bar', uuid='blag')
        >>> list(self._open().items())
        >>> print(self.read())
        >>> print(self.pandas())
        >>> print(self.fpath.stat().st_size)
    """

    def append(self, **kwargs):
        unknown = ub.udict(kwargs) - self.__default__
        if unknown:
            raise ValueError(f'Unknown kwargs={unknown}')
        row = ub.udict(self.__default__)
        row |= (kwargs & row)
        shelf = self._open()
        try:
            if '__list__' not in shelf:
                shelf['__list__'] = []
            accum = shelf['__list__']
            accum.append(row)
            # Inefficient, can split across multiple entries giving each a max
            # size.
            shelf['__list__'] = accum
        finally:
            shelf.close()

    def read(self):
        registry_rows = []
        shelf = self._open()
        try:
            registry_rows += list(shelf['__list__'])
        except Exception as ex:
            warnings.warn('Unable to access shelf: {}'.format(ex))
        finally:
            shelf.close()
        print(f'registry_rows = {ub.urepr(registry_rows, nl=1)}')
        return registry_rows

    def pandas(self, **kwargs):
        import pandas as pd
        df = pd.DataFrame(self.query(**kwargs))
        return df


class NewDVCRegistery(DictRegistery):
    # TODO:
    __registery_fname__ = 'geowatch_new_dvc_registery.shelf'

    __default__ = ub.udict({
        'name': ub.NoParam,
        'path': ub.NoParam,
        'priority': None,
        'hardware': None,
        'tags': None,
    })


class FitRegistery(ListRegistery):
    """
    This is a shelf with a single list of rows that has a globally consistent
    path on each machine. The __default__ attribute specifies which keys each
    row will have. This is used to store information whenever training is run
    and stores the path to its output directory. This can be used to inspect
    which training runs artifacts might be available on this machine and also
    to help the user navigate to recent paths.

    Example:
        >>> from geowatch.tasks.fusion.fit_registery import *  # NOQA
        >>> class DemoFitRegistery(FitRegistery):
        >>>     __registery_fname__ = 'geowatch_demo_fit_registery.shelf'
        >>>     __appname__ = 'geowatch/demo'
        >>> #
        >>> self = DemoFitRegistery()
        >>> self.append(path=ub.Path('~').expand())
        >>> self.append(path=ub.Path('~').expand())
        >>> self.append(path=ub.Path('~').expand())
        >>> print(self.pandas())
        >>> print(self._items())

    Example:
        >>> # This is the real one, dont modify it
        >>> from geowatch.tasks.fusion.fit_registery import *  # NOQA
        >>> self = FitRegistery()
        >>> print(self.pandas())
        >>> print(self._items())
    """
    __registery_fname__ = 'geowatch_fit_registery.shelf'
    __default__ = ub.udict({
        'path'     : ub.NoParam,
        'uuid'     : None,
        'time'     : None,
        'user'     : None,
        'hostname' : None,
        'argv'     : None,
        'config'   : None,
    })

    def append(self, **kwargs):
        import os
        import uuid
        import platform
        path = kwargs.get('path', None)
        if path is not None:
            kwargs['path'] = os.fspath(path)

        path = kwargs.get('uuid', None)
        if path is None:
            kwargs['uuid'] = str(uuid.uuid4())

        time = kwargs.get('time', None)
        if time is None:
            kwargs['time'] = ub.timestamp()

        user = kwargs.get('user', None)
        if user is None:
            kwargs['user'] = ub.Path.home().name

        hostname = kwargs.get('hostname', None)
        if hostname is None:
            kwargs['hostname'] = platform.node()

        super().append(**kwargs)

    def peek(self):
        items = self.read()
        if not items:
            return None
        else:
            return items[-1]


class FitRegisteryCLI(scfg.ModalCLI):
    """
    Command line helper
    """
    __command__ = 'registery'

    class Append(scfg.DataConfig):
        """
        Register a new path (or overwrite / update an existing one)
        """
        __command__ = 'add'
        path = scfg.Value(None, help='path to register', position=1)

        @classmethod
        def main(cls, cmdline=1, **kwargs):
            config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
            registry = FitRegistery()
            registry.append(**config)

    class List(scfg.DataConfig):
        """
        List registered paths
        """
        __command__ = 'list'

        @classmethod
        def main(cls, cmdline=1, **kwargs):
            config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
            config = dict(config)
            registry = FitRegistery()
            registry.list(**config)

    class Peek(scfg.DataConfig):
        """
        Search for a path registered via ``sdvc registry add``
        """
        __command__ = 'peek'

        @classmethod
        def main(cls, cmdline=1, **kwargs):
            cls.cli(cmdline=cmdline, data=kwargs, strict=True)
            registry = FitRegistery()
            row = registry.peek()
            print(f'row = {ub.urepr(row, nl=1)}')


if __name__ == '__main__':
    """
    CommandLine:
        python -m geowatch.tasks.fusion.fit_registery peek
    """
    FitRegisteryCLI.main()
