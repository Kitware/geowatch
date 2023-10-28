#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CliFormatterCLI(scfg.DataConfig):
    """
    The idea is that we can ingest a dictionary, argv list, or a command line
    string and convert between any of these formats.
    """
    input = scfg.Value(None, type=str, help='the input', position=1)

    input_type = scfg.Value('infer', help='attempt to infer what type the input is')

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

        config_dict = eval(config.input, {}, {})
        # input_type = 'config_dict'
        clistr = make_argstr(config_dict)
        print(clistr)

        # from kwutil import util_yaml
        # print(util_yaml.Yaml.coerce(config.input))


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


__cli__ = CliFormatterCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/poc/cli_formatter.py
        python -m cli_formatter
    """
    main()
