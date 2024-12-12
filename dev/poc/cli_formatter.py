#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CliFormatterCLI(scfg.DataConfig):
    """
    The idea is that we can ingest a dictionary, argv list, or a command line
    string and convert between any of these formats.
    """
    input = scfg.Value(None, type=str, help='the input', position=1)

    input_type = scfg.Value('auto', help='attempt to infer what type the input is')

    output_type = scfg.Value('all', help='output type to convert to.')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from cli_formatter import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = CliFormatterCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))
        import kwutil

        input_type = config.input_type
        text = config.input
        if input_type == 'auto':
            input_type = guess_input_type(input_type)
            print(f'Guess: input_type={input_type}')

        config_dict = parse_cli_input(text, input_type)

        if config.output_type == 'all':
            output_types = ['yaml', 'dict', 'argv']
        else:
            output_types = config.output_type

        for output_type in output_types:
            if output_type == 'dict':
                print(config_dict)
            if output_type == 'yaml':
                print(kwutil.Yaml.dumps(config_dict))
            if output_type == 'argv':
                clistr = make_argstr(config_dict)
                print(clistr)

        # from kwutil import util_yaml
        # print(util_yaml.Yaml.coerce(config.input))


def parse_cli_input(text, input_type='auto'):
    """
    CommandLine:
        xdoctest -m /home/joncrall/code/geowatch/dev/poc/cli_formatter.py parse_cli_input
        xdoctest -m cli_formatter parse_cli_input

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/geowatch/dev/poc'))
        >>> from cli_formatter import *  # NOQA
        >>> cases = [
        >>>      '--foo=bar --bar',
        >>>     '{foo: bar, baz: true}',
        >>>     '{"foo": "bar", "baz": True}'
        >>> ]
        >>> for text in cases:
        >>>     print('--- Case ---')
        >>>     print(f' * text = {ub.urepr(text, nl=1)}')
        >>>     input_type = guess_input_type(text)
        >>>     print(f' * input_type = {ub.urepr(input_type, nl=1)}')
        >>>     config_dict = parse_cli_input(text)
        >>>     print(f' * config_dict = {ub.urepr(config_dict, nl=1)}')
    """
    import kwutil
    if input_type == 'auto':
        input_type = guess_input_type(text)
    if input_type == 'dict':
        config_dict = eval(text, {}, {})
    elif input_type == 'yaml':
        config_dict = kwutil.Yaml.coerce(text)
    elif input_type == 'argv':
        config_dict = parse_argv_as_config_dict(text)
    else:
        raise KeyError(input_type)
    return config_dict


def score_cli_invocation(text):
    """
    References:
        https://chatgpt.com/c/675b37b7-3fe8-8002-a3a8-845e0cb6c200

    Example:
        cases = [
            '{foo: bar}',
            '--key=value',
            '--key=value --key value',
            '--key value',
            '--key value pos arg'
            '--key value -- position argument',
        ]
        for text in cases:
            print(score_cli_invocation(text))
    """
    import re
    # Patterns for common CLI elements
    flag_pattern = r'--\w+(?:=\S+)?(?:\s+\S+)?'  # Match flags with or without values
    positional_pattern = r'(?:--\s|--\w+(?:=\S+|\s+\S+))\s*(.*)'  # Match positional args after "--"

    # Scoring factors
    score = 0

    # Count valid flags
    flags = re.findall(flag_pattern, text)
    score += len(flags)  # Each valid flag adds points

    # Check for positional arguments
    found = re.search(positional_pattern, text)
    if found:
        score += len(found.groups()) * 2  # Each valid flag adds points
    return score


def input_type_scores(text):
    """
    Score liklihood of each:
        * A python dictionary
        * A YAML dictionary
        * An argv string

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/geowatch/dev/poc'))
        >>> from cli_formatter import *  # NOQA
        >>> cases = [
        >>>      '--foo=bar',
        >>>     'foo: bar',
        >>>     '{"foo": "bar"}'
        >>> ]
        >>> for text in cases:
        >>>     scores = input_type_scores(text)
        >>>     print(f'{text!r} - {scores}')
    """
    import kwutil
    scores = {
        'yaml': 0,
        'dict': 0,
        'argv': 0,
    }
    try:
        result = kwutil.Yaml.coerce(text)
        if isinstance(result, dict):
            scores['yaml'] += 2
    except Exception:
        ...

    try:
        if 0:
            # unsafe
            result = eval(text, {}, {})
        else:
            # safer
            result = kwutil.safeeval(text, addnodes=['Dict', 'List'])
        if isinstance(result, dict):
            scores['dict'] += 3
    except Exception:
        ...

    try:
        scores['argv'] = score_cli_invocation(text)
    except Exception:
        ...
    return scores


def guess_input_type(text):
    """
    Try and figure out if this is:
        * A python dictionary
        * A YAML dictionary
        * An argv string

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/geowatch/dev/poc'))
        >>> from cli_formatter import *  # NOQA
        >>> cases = [
        >>>      '--foo=bar',
        >>>     'foo: bar',
        >>>     '{"foo": "bar"}'
        >>> ]
        >>> for text in cases:
        >>>     guess = guess_input_type(text)
        >>>     print(f'{text!r} - {guess}')
    """
    scores = input_type_scores(text)
    guess = ub.argmax(scores)
    return guess


def make_argstr(config_dict):
    # parts = [f'    --{k}="{v}" \\' for k, v in config.items()]
    parts = []
    import shlex
    for k, v in config_dict.items():
        if isinstance(v, list):
            # Handle variable-args params
            quoted_varargs = [shlex.quote(str(x)) for x in v]
            preped_varargs = ['        ' + x + ' \\' for x in quoted_varargs]
            parts.append(f'    --{k} \\')
            parts.extend(preped_varargs)
        else:
            import shlex
            vstr = shlex.quote(str(v))
            parts.append(f'    --{k}={vstr} \\')

    clistr = '\n'.join(parts).lstrip().rstrip('\\')
    return clistr


def pretty_argjoin(args):
    import shlex
    max_width = 80
    lines = []
    current_line = ''
    for arg in args:
        is_argkey = arg.startswith('--')

        arg = shlex.quote(arg)
        if is_argkey or len(current_line) > max_width:
            # Accept current lines
            lines.append(current_line)
            current_line = ''

        if len(current_line) == 0:
            if len(lines):
                current_line += ('    ' + arg)
            else:
                current_line += arg
        else:
            current_line += ' '
            current_line += arg

    if current_line:
        lines.append(current_line)

    text = '\n'.join([line + ' \\' for line in lines])
    text = text.strip('\\').strip(' ')
    return text


def parse_argv_as_config_dict(text):
    """
    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/geowatch/dev/poc'))
        >>> from cli_formatter import *  # NOQA
        >>> text = '--foo=bar'
        >>> config = parse_argv_as_config_dict(text)
        >>> print(f'config = {ub.urepr(config, nl=1)}')
        >>> #
        >>> text = '--foo=bar --bar 1 --baz'
        >>> config = parse_argv_as_config_dict(text)
        >>> print(f'config = {ub.urepr(config, nl=1)}')
        >>> #
        >>> text = '--foo bar baz biz --key=value --flag1 --flag2 -- pos1 pos2'
        >>> config = parse_argv_as_config_dict(text)
        >>> print(f'config = {ub.urepr(config, nl=1)}')
    """
    components = parse_bash_invocation(text)
    config = {}
    for item in components:
        if item['type'] == 'positional':
            continue
        if item['type'] == 'separator':
            # no more key/value to parse
            break
        if item['type'].startswith('keyvalue'):
            key = item['key']
            value = item['value']
            assert key not in config
            config[key] = value
        if item['type'].startswith('flag'):
            key = item['key']
            value = item['value']
            assert key not in config
            config[key] = value
    return config


def parse_bash_invocation(bash_text, with_tokens=False):
    r"""
    Modified from ChatGPT response

    SeeAlso:
        bashlex - https://pypi.org/project/bashlex/

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/geowatch/dev/poc'))
        >>> from cli_formatter import *  # NOQA
        >>> bash_text = r'''
            # Leading comment
            python -m geowatch.tasks.fusion.evaluate \
                --pred_dataset=pred.kwcoco.zip \
                --true_dataset=/test_imgs30_d8988f8c.kwcoco.zip \
                --eval_dpath=/heatmap_eval/heatmap_eval_id_d0acafbb \
                --age=30 --verbose \
                -o output.txt positional1 -- all rest are positions \
                positional2 --even me --and=me
        >>> '''
        >>> components = parse_bash_invocation(bash_text)
        >>> import pandas as pd
        >>> pd.DataFrame(components)
        >>> components = parse_bash_invocation(bash_text, with_tokens=True)
        >>> pd.DataFrame(components)

    Example:
        >>> text = '--foo bar baz biz --key=value --subkey "{big: long, yaml: text--text}" --flag -- pos1 pos2'
        >>> components = parse_bash_invocation(text, with_tokens=True)
        >>> print(f'components = {ub.urepr(components, nl=1)}')
    """
    import re
    # Split the bash_text into tokens based on spaces, keeping the structure intact
    import bashlex
    tokens = list(bashlex.split(bash_text.strip()))
    # import shlex
    # bash_text = bash_text.replace('\\\n', ' ')
    # tokens = shlex.split(bash_text)
    # tokens = bash_text.split()
    components = []
    index = 0

    found_start = False
    position_only_mode = False

    while index < len(tokens):
        token = tokens[index]

        if not found_start:
            if token.strip():
                found_start = True
            else:
                # Skip leading blank tokens
                index += 1
                continue

        if token == '--':
            position_only_mode = True
            index += 1
            if with_tokens:
                item = {
                    'type': 'separator',
                    'token_index_start': index,
                    'token_index_stop': index + 1,
                    'tokens': tokens[index: index + 1]
                }
                components.append(item)
            continue

        handled = False
        if not position_only_mode:
            # Handle --key=value or -key=value
            if re.match(r'--?\w+=', token):
                key, value = token.split('=', 1)
                param_name = key.lstrip('-')
                # args_dict[param_name] = value
                item = {
                    'type': 'keyvalue_eq',
                    'key': param_name,
                    'value': value,
                }
                if with_tokens:
                    item['token_index_start'] = index
                    item['token_index_stop'] = index + 1
                    item['tokens'] = tokens[index: index + 1]
                handled = True

            # Handle --key value or -key value
            elif token.startswith('--') or token.startswith('-'):
                key = token.lstrip('-')
                token_index_start = index
                next_values = []
                while index + 1 < len(tokens) and not tokens[index + 1].startswith('-'):
                    value = tokens[index + 1]
                    next_values.append(value)
                    index += 1
                if len(next_values) == 0:
                    # No values, must be a flag
                    item = {
                        'type': 'flag',
                        'key': key,
                        'value': True,
                    }
                    if with_tokens:
                        item['token_index_start'] = index
                        item['token_index_stop'] = index + 1
                        item['tokens'] = tokens[index: index + 1]

                else:
                    if len(next_values) == 1:
                        value = next_values[0]
                    else:
                        value = next_values
                    item = {
                        'type': 'keyvalue_space',
                        'key': key,
                        'value': value,
                    }
                    token_index_stop = index + 1
                    if with_tokens:
                        item['token_index_start'] = token_index_start
                        item['token_index_stop'] = token_index_stop
                        item['tokens'] = tokens[token_index_start: token_index_stop]
                handled = True

        # Handle positional arguments
        if not handled:
            item = {
                'type': 'positional',
                'value': token,
            }
            if with_tokens:
                item['token_index_start'] = index
                item['token_index_stop'] = index + 1
                item['tokens'] = tokens[index: index + 1]
        components.append(item)
        index += 1
    return components


__cli__ = CliFormatterCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/poc/cli_formatter.py
        python -m cli_formatter
    """
    main()
