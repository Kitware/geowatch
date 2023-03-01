import io
import os
import ubelt as ub


def yaml_dumps(data):
    """
    Dump yaml to a string representation
    (and account for some of our use-cases)

    Args:
        data (Any): yaml representable data

    Returns:
        str: yaml text

    Example:
        >>> from watch.utils.util_yaml import *  # NOQA
        >>> import ubelt as ub
        >>> data = {
        >>>     'a': 'hello world',
        >>>     'b': ub.udict({'a': 3})
        >>> }
        >>> text = yaml_dumps(data)
        >>> print(text)
    """
    import yaml

    def str_presenter(dumper, data):
        # https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        if len(data.splitlines()) > 1 or '\n' in data:
            text_list = [line.rstrip() for line in data.splitlines()]
            fixed_data = '\n'.join(text_list)
            return dumper.represent_scalar('tag:yaml.org,2002:str', fixed_data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    class Dumper(yaml.Dumper):
        pass

    Dumper.add_representer(str, str_presenter)
    Dumper.add_representer(ub.udict, Dumper.represent_dict)
    # dumper = yaml.dumper.Dumper
    # dumper = yaml.SafeDumper(sort_keys=False)
    # yaml.dump(data, s, Dumper=yaml.SafeDumper, sort_keys=False, width=float("inf"))
    s = io.StringIO()
    # yaml.dump(data, s, sort_keys=False)
    yaml.dump(data, s, Dumper=Dumper, sort_keys=False, width=float("inf"))
    s.seek(0)
    text = s.read()
    return text


def yaml_loads(text, backend='ruamel'):
    """
    Load yaml from a text

    Args:
        text (str): yaml text

    Returns:
        object

    Example:
        >>> from watch.utils import util_yaml
        >>> import ubelt as ub
        >>> data = {
        >>>     'a': 'hello world',
        >>>     'b': ub.udict({'a': 3})
        >>> }
        >>> print('data = {}'.format(ub.urepr(data, nl=1)))
        >>> print('---')
        >>> text = util_yaml.yaml_dumps(data)
        >>> print(ub.highlight_code(text, 'yaml'))
        >>> print('---')
        >>> data2 = util_yaml.yaml_loads(text)
        >>> assert data == data2
        >>> data3 = util_yaml.yaml_loads(text, backend='pyyaml')
        >>> print('data2 = {}'.format(ub.urepr(data2, nl=1)))
        >>> print('data3 = {}'.format(ub.urepr(data3, nl=1)))
        >>> assert data == data3
    """
    file = io.StringIO(text)
    # data = yaml.load(file, Loader=yaml.SafeLoader)
    if backend == 'ruamel':
        import ruamel.yaml
        data = ruamel.yaml.load(file, Loader=ruamel.yaml.RoundTripLoader, preserve_quotes=True)
    elif backend == 'pyyaml':
        import yaml
        # data = yaml.load(file, Loader=yaml.SafeLoader)
        data = yaml.load(file, Loader=yaml.Loader)
    else:
        raise KeyError(backend)
    return data


def yaml_load(file, backend='ruamel'):
    """
    Load yaml from a file

    Args:
        file (io.TextIO | PathLike | str): yaml file path or file object

    Returns:
        object
    """
    if isinstance(file, (str, os.PathLike)):
        with open(file, 'r') as fp:
            return yaml_load(fp, backend=backend)
    else:
        if backend == 'ruamel':
            import ruamel.yaml
            data = ruamel.yaml.load(file, Loader=ruamel.yaml.RoundTripLoader, preserve_quotes=True)
        elif backend == 'pyyaml':
            import yaml
            data = yaml.load(file, Loader=yaml.Loader)
        else:
            raise KeyError(backend)
        return data


def coerce_yaml(data, backend='ruamel'):
    """
    Attempt to convert input into a parsed yaml / json data structure.
    If the data looks like a path, it tries to load and parse file contents.
    If the data looks like a yaml/json string it tries to parse it.
    If the data looks like parsed data, then it returns it as-is.

    Args:
        data (str | PathLike | dict | list):

    Returns:
        object: parsed yaml data

    Note:
        The input to the function cannot distinguish a string that should be
        loaded and a string that should be parsed. If it looks like a file that
        exists it will read it. To avoid this coerner case use this only for
        data where you expect the output is a List or Dict.

    Example:
        >>> from watch.utils.util_yaml import *  # NOQA
        >>> coerce_yaml('"[1, 2, 3]"')
        [1, 2, 3]
        >>> fpath = ub.Path.appdir('watch/tests/util_yaml').ensuredir() / 'file.yaml'
        >>> fpath.write_text(yaml_dumps([4, 5, 6]))
        >>> coerce_yaml(fpath)
        [4, 5, 6]
        >>> coerce_yaml(str(fpath))
        [4, 5, 6]
        >>> dict(coerce_yaml('{a: b, c: d}'))
        {'a': 'b', 'c': 'd'}
        >>> coerce_yaml(None)
        None
    """
    if isinstance(data, str):
        maybe_path = None
        if '\n' not in data:
            # Ambiguous case: might this be path-like?
            maybe_path = ub.Path(data)
            if not maybe_path.exists():
                maybe_path = None

        if maybe_path is not None:
            result = coerce_yaml(maybe_path, backend=backend)
        else:
            result = yaml_loads(data, backend=backend)
    elif isinstance(data, os.PathLike):
        result = yaml_load(data, backend=backend)
    elif hasattr(data, 'read'):
        # assume file
        result = yaml_load(data, backend=backend)
    else:
        # Probably already parsed. Return the input
        result = data
    return result
