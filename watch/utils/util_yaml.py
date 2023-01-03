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


def yaml_loads(text):
    import io
    file = io.StringIO(text)
    # data = yaml.load(file, Loader=yaml.SafeLoader)
    import ruamel.yaml
    data = ruamel.yaml.load(file, Loader=ruamel.yaml.RoundTripLoader, preserve_quotes=True)
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
