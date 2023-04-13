import io
import os
import ubelt as ub


class _YamlRepresenter:

    @staticmethod
    def str_presenter(dumper, data):
        # https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        if len(data.splitlines()) > 1 or '\n' in data:
            text_list = [line.rstrip() for line in data.splitlines()]
            fixed_data = '\n'.join(text_list)
            return dumper.represent_scalar('tag:yaml.org,2002:str', fixed_data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)


@ub.memoize
def _custom_ruaml_loader():
    """
    References:
        https://stackoverflow.com/questions/59635900/ruamel-yaml-custom-commentedmapping-for-custom-tags
        https://stackoverflow.com/questions/528281/how-can-i-include-a-yaml-file-inside-another
    """
    import ruamel.yaml
    Loader = ruamel.yaml.RoundTripLoader

    def _construct_include_tag(self, node):
        print(f'node={node}')
        if isinstance(node.value, list):
            return [Yaml.coerce(v.value) for v in node.value]
        else:
            external_fpath = ub.Path(node.value)
            if not external_fpath.exists():
                raise IOError(f'Included external yaml file {external_fpath} '
                              'does not exist')
            return Yaml.load(node.value)
    Loader.add_constructor("!include", _construct_include_tag)
    return Loader


@ub.memoize
def _custom_ruaml_dumper():
    """
    References:
        https://stackoverflow.com/questions/59635900/ruamel-yaml-custom-commentedmapping-for-custom-tags
    """
    import ruamel.yaml
    Dumper = ruamel.yaml.RoundTripDumper
    Dumper.add_representer(str, _YamlRepresenter.str_presenter)
    Dumper.add_representer(ub.udict, Dumper.represent_dict)
    return Dumper


@ub.memoize
def _custom_pyaml_dumper():
    import yaml

    class Dumper(yaml.Dumper):
        pass
    # dumper = yaml.dumper.Dumper
    # dumper = yaml.SafeDumper(sort_keys=False)
    # yaml.dump(data, s, Dumper=yaml.SafeDumper, sort_keys=False, width=float("inf"))
    # yaml.dump(data, s, sort_keys=False)
    Dumper.add_representer(str, _YamlRepresenter.str_presenter)
    Dumper.add_representer(ub.udict, Dumper.represent_dict)
    return Dumper


class Yaml:
    """
    Namespace for yaml functions
    """

    @staticmethod
    def dumps(data, backend='ruamel'):
        """
        Dump yaml to a string representation
        (and account for some of our use-cases)

        Args:
            data (Any): yaml representable data
            backend (str): either ruamel or pyyaml

        Returns:
            str: yaml text

        Example:
            >>> import ubelt as ub
            >>> data = {
            >>>     'a': 'hello world',
            >>>     'b': ub.udict({'a': 3})
            >>> }
            >>> text1 = Yaml.dumps(data, backend='ruamel')
            >>> print(text1)
            >>> text2 = Yaml.dumps(data, backend='pyyaml')
            >>> print(text2)
            >>> assert text1 == text2
        """
        file = io.StringIO()
        if backend == 'ruamel':
            import ruamel.yaml
            Dumper = _custom_ruaml_dumper()
            ruamel.yaml.round_trip_dump(data, file, Dumper=Dumper, width=float("inf"))
        elif backend == 'pyyaml':
            import yaml
            Dumper = _custom_pyaml_dumper()
            yaml.dump(data, file, Dumper=Dumper, sort_keys=False, width=float("inf"))
        else:
            raise KeyError(backend)
        text = file.getvalue()
        return text

    @staticmethod
    def load(file, backend='ruamel'):
        """
        Load yaml from a file

        Args:
            file (io.TextIO | PathLike | str): yaml file path or file object
            backend (str): either ruamel or pyyaml

        Returns:
            object
        """
        if isinstance(file, (str, os.PathLike)):
            with open(file, 'r') as fp:
                return Yaml.load(fp, backend=backend)
        else:
            if backend == 'ruamel':
                import ruamel.yaml
                Loader = _custom_ruaml_loader()
                data = ruamel.yaml.load(file, Loader=Loader, preserve_quotes=True)
                # data = ruamel.yaml.load(file, Loader=ruamel.yaml.RoundTripLoader, preserve_quotes=True)
            elif backend == 'pyyaml':
                import yaml
                # data = yaml.load(file, Loader=yaml.SafeLoader)
                data = yaml.load(file, Loader=yaml.Loader)
            else:
                raise KeyError(backend)
            return data

    @staticmethod
    def loads(text, backend='ruamel'):
        """
        Load yaml from a text

        Args:
            text (str): yaml text
            backend (str): either ruamel or pyyaml

        Returns:
            object

        Example:
            >>> import ubelt as ub
            >>> data = {
            >>>     'a': 'hello world',
            >>>     'b': ub.udict({'a': 3})
            >>> }
            >>> print('data = {}'.format(ub.urepr(data, nl=1)))
            >>> print('---')
            >>> text = Yaml.dumps(data)
            >>> print(ub.highlight_code(text, 'yaml'))
            >>> print('---')
            >>> data2 = Yaml.loads(text)
            >>> assert data == data2
            >>> data3 = Yaml.loads(text, backend='pyyaml')
            >>> print('data2 = {}'.format(ub.urepr(data2, nl=1)))
            >>> print('data3 = {}'.format(ub.urepr(data3, nl=1)))
            >>> assert data == data3
        """
        file = io.StringIO(text)
        return Yaml.load(file, backend=backend)

    @staticmethod
    def coerce(data, backend='ruamel'):
        """
        Attempt to convert input into a parsed yaml / json data structure.
        If the data looks like a path, it tries to load and parse file contents.
        If the data looks like a yaml/json string it tries to parse it.
        If the data looks like parsed data, then it returns it as-is.

        Args:
            data (str | PathLike | dict | list):
            backend (str): either ruamel or pyyaml

        Returns:
            object: parsed yaml data

        Note:
            The input to the function cannot distinguish a string that should be
            loaded and a string that should be parsed. If it looks like a file that
            exists it will read it. To avoid this coerner case use this only for
            data where you expect the output is a List or Dict.

        References:
            https://stackoverflow.com/questions/528281/how-can-i-include-a-yaml-file-inside-another

        Example:
            >>> Yaml.coerce('"[1, 2, 3]"')
            [1, 2, 3]
            >>> fpath = ub.Path.appdir('cmd_queue/tests/util_yaml').ensuredir() / 'file.yaml'
            >>> fpath.write_text(Yaml.dumps([4, 5, 6]))
            >>> Yaml.coerce(fpath)
            [4, 5, 6]
            >>> Yaml.coerce(str(fpath))
            [4, 5, 6]
            >>> dict(Yaml.coerce('{a: b, c: d}'))
            {'a': 'b', 'c': 'd'}
            >>> Yaml.coerce(None)
            None

        Example:
            >>> from watch.utils.util_yaml import *  # NOQA
            >>> assert Yaml.coerce('') is None

        Example:
            >>> dpath = ub.Path.appdir('cmd_queue/tests/util_yaml').ensuredir()
            >>> fpath = dpath / 'external.yaml'
            >>> fpath.write_text(Yaml.dumps({'foo': 'bar'}))
            >>> text = ub.codeblock(
            >>>    f'''
            >>>    items:
            >>>        - !include {dpath}/external.yaml
            >>>    ''')
            >>> data = Yaml.coerce(text, backend='ruamel')
            >>> print(Yaml.dumps(data, backend='ruamel'))
            items:
            - foo: bar

            >>> text = ub.codeblock(
            >>>    f'''
            >>>    items:
            >>>        !include [{dpath}/external.yaml, blah, 1, 2, 3]
            >>>    ''')
            >>> data = Yaml.coerce(text, backend='ruamel')
            >>> print('data = {}'.format(ub.urepr(data, nl=1)))
            >>> print(Yaml.dumps(data, backend='ruamel'))
        """
        if isinstance(data, str):
            maybe_path = None
            if '\n' not in data and len(data.strip()) > 0:
                # Ambiguous case: might this be path-like?
                maybe_path = ub.Path(data)
                try:
                    if not maybe_path.exists():
                        maybe_path = None
                except OSError:
                    maybe_path = None
            if maybe_path is not None:
                result = Yaml.coerce(maybe_path, backend=backend)
            else:
                result = Yaml.loads(data, backend=backend)
        elif isinstance(data, os.PathLike):
            result = Yaml.load(data, backend=backend)
        elif hasattr(data, 'read'):
            # assume file
            result = Yaml.load(data, backend=backend)
        else:
            # Probably already parsed. Return the input
            result = data
        return result

    @staticmethod
    def InlineList(items):
        """
        References:
            .. [SO56937691] https://stackoverflow.com/questions/56937691/making-yaml-ruamel-yaml-always-dump-lists-inline
        """
        import ruamel.yaml
        ret = ruamel.yaml.comments.CommentedSeq(items)
        ret.fa.set_flow_style()
        return ret
