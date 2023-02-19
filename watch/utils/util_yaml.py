def yaml_dumps(data):
    """
    CommandLine:
        xdoctest -m watch.utils.util_yaml yaml_dumps

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
    import io
    import ubelt as ub

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
    import io
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


def yaml_load(file):
    import os
    if isinstance(file, (str, os.PathLike)):
        with open(file, 'r') as fp:
            return yaml_load(fp)
    else:
        import ruamel.yaml
        data = ruamel.yaml.load(file, Loader=ruamel.yaml.RoundTripLoader, preserve_quotes=True)
        return data
