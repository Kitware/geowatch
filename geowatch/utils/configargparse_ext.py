import configargparse
import argparse
import sys
import os
import re
import ubelt as ub
from collections import OrderedDict

from configargparse import (
    already_on_command_line,
    _COMMAND_LINE_SOURCE_KEY,
    OPTIONAL, ZERO_OR_MORE,
    SUPPRESS,
    ACTION_TYPES_THAT_DONT_NEED_A_VALUE,
    _CONFIG_FILE_SOURCE_KEY, _DEFAULTS_SOURCE_KEY, ConfigFileParserException,
    _ENV_VAR_SOURCE_KEY, StringIO
)


class RawDescriptionDefaultsHelpFormatter(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter):
    pass


class ArgumentParser(configargparse.ArgumentParser):
    """
    Todo: maybe submit these as new features to configargparse
    """

    def __init__(self, *args, **kwargs):
        """
        Adds additional ability to handle special string settings

        New keys:
            formatter_class : can be "raw"
            config_file_parser_class : can be "yaml"
        """

        special_defaults = {
            'formatter_class': {
                'raw': configargparse.ArgumentDefaultsRawHelpFormatter,
                'defaults': configargparse.ArgumentDefaultsHelpFormatter,
            },
            'config_file_parser_class': {
                'yaml': configargparse.YAMLConfigFileParser,
            },
        }
        # For each item, replace any special string keys with a standard value
        for key, defaults in special_defaults.items():
            val = kwargs.get(key, None)
            if isinstance(val, str):
                if val in defaults:
                    kwargs[key] = defaults[val]
                else:
                    raise KeyError('Unknown setting {} for {}'.format(val, key))
        # import ubelt as ub
        # print('kwargs = {}'.format(ub.urepr(kwargs, nl=1)))
        super().__init__(*args, **kwargs)

    # Annoying that I have to pull in all of this code
    # SeeAlso: https://github.com/bw2/ConfigArgParse/pull/246
    def parse_known_args(
            self,
            args=None,
            namespace=None,
            config_file_contents=None,
            env_vars=ub.NoParam,
            ignore_help_args=False,
            ignore_write_args=False,
    ):
        """Supports all the same args as the ArgumentParser.parse_args(..),
        as well as the following additional args.

        Arguments:
            args: a list of args as in argparse, or a string (eg. "-x -y bla")
            config_file_contents (str). Used for testing.
            env_vars (dict). Used for testing.
            ignore_help_args (bool): This flag determines behavior when user specifies --help or -h. If False,
                it will have the default behavior - printing help and exiting. If True, it won't do either.
        """
        if env_vars is ub.NoParam:
            env_vars = os.environ

        if args is None:
            args = sys.argv[1:]
        elif isinstance(args, str):
            args = args.split()
        else:
            args = list(args)

        for a in self._actions:
            a.is_positional_arg = not a.option_strings

        if ignore_help_args:
            args = [arg for arg in args if arg not in ('-h', '--help')]

        # maps a string describing the source (eg. env var) to a settings dict
        # to keep track of where values came from (used by print_values()).
        # The settings dicts for env vars and config files will then map
        # the config key to an (argparse Action obj, string value) 2-tuple.
        self._source_to_settings = OrderedDict()
        if args:
            a_v_pair = (None, list(args))  # copy args list to isolate changes
            self._source_to_settings[_COMMAND_LINE_SOURCE_KEY] = {'': a_v_pair}

        # handle auto_env_var_prefix __init__ arg by setting a.env_var as needed
        if self._auto_env_var_prefix is not None:
            for a in self._actions:
                config_file_keys = self.get_possible_config_keys(a)
                if (config_file_keys and not (a.env_var or a.is_positional_arg
                                              or a.is_config_file_arg or a.is_write_out_config_file_arg or
                                              isinstance(a, argparse._VersionAction) or
                                              isinstance(a, argparse._HelpAction))):
                    stripped_config_file_key = config_file_keys[0].strip(
                        self.prefix_chars)
                    a.env_var = (self._auto_env_var_prefix +
                                 stripped_config_file_key).replace('-', '_').upper()

        # add env var settings to the commandline that aren't there already
        env_var_args = []
        nargs = False
        actions_with_env_var_values = [a for a in self._actions
                                       if not a.is_positional_arg and a.env_var and a.env_var in env_vars
                                       and not already_on_command_line(args, a.option_strings, self.prefix_chars)]
        for action in actions_with_env_var_values:
            key = action.env_var
            value = env_vars[key]
            # Make list-string into list.
            if action.nargs or isinstance(action, argparse._AppendAction):
                nargs = True
                element_capture = re.match(r'\[(.*)\]', value)
                if element_capture:
                    value = [val.strip() for val in element_capture.group(1).split(',') if val.strip()]
            env_var_args += self.convert_item_to_command_line_arg(
                action, key, value)

        if nargs:
            args = args + env_var_args
        else:
            args = env_var_args + args

        if env_var_args:
            self._source_to_settings[_ENV_VAR_SOURCE_KEY] = OrderedDict(
                [(a.env_var, (a, env_vars[a.env_var]))
                    for a in actions_with_env_var_values])

        # before parsing any config files, check if -h was specified.
        supports_help_arg = any(
            a for a in self._actions if isinstance(a, argparse._HelpAction))
        skip_config_file_parsing = supports_help_arg and (
            '-h' in args or '--help' in args)

        # prepare for reading config file(s)
        known_config_keys = {config_key: action for action in self._actions
                             for config_key in self.get_possible_config_keys(action)}

        # open the config file(s)
        config_streams = []
        if config_file_contents is not None:
            stream = StringIO(config_file_contents)
            stream.name = 'method arg'
            config_streams = [stream]
        elif not skip_config_file_parsing:
            config_streams = self._open_config_files(args)

        # parse each config file
        for stream in reversed(config_streams):
            try:
                config_items = self._config_file_parser.parse(stream)
            except ConfigFileParserException as e:
                self.error(e)
            finally:
                if hasattr(stream, 'close'):
                    stream.close()

            # add each config item to the commandline unless it's there already
            config_args = []
            nargs = False
            for key, value in config_items.items():
                if key in known_config_keys:
                    action = known_config_keys[key]
                    discard_this_key = already_on_command_line(
                        args, action.option_strings, self.prefix_chars)
                else:
                    action = None
                    discard_this_key = self._ignore_unknown_config_file_keys or \
                        already_on_command_line(
                            args,
                            [self.get_command_line_key_for_unknown_config_file_setting(key)],
                            self.prefix_chars)

                if not discard_this_key:
                    config_args += self.convert_item_to_command_line_arg(
                        action, key, value)
                    source_key = '%s|%s' % (_CONFIG_FILE_SOURCE_KEY, stream.name)
                    if source_key not in self._source_to_settings:
                        self._source_to_settings[source_key] = OrderedDict()
                    self._source_to_settings[source_key][key] = (action, value)
                    if (action and action.nargs or isinstance(action, argparse._AppendAction)):
                        nargs = True

            if nargs:
                args = args + config_args
            else:
                args = config_args + args

        # save default settings for use by print_values()
        default_settings = OrderedDict()
        for action in self._actions:
            cares_about_default_value = (not action.is_positional_arg or action.nargs in [OPTIONAL, ZERO_OR_MORE])
            if (already_on_command_line(args, action.option_strings, self.prefix_chars) or
                    not cares_about_default_value or
                    action.default is None or
                    action.default == SUPPRESS or
                    isinstance(action, ACTION_TYPES_THAT_DONT_NEED_A_VALUE)):
                continue
            else:
                if action.option_strings:
                    key = action.option_strings[-1]
                else:
                    key = action.dest
                default_settings[key] = (action, str(action.default))

        if default_settings:
            self._source_to_settings[_DEFAULTS_SOURCE_KEY] = default_settings

        # parse all args (including commandline, config file, and env var)
        namespace, unknown_args = argparse.ArgumentParser.parse_known_args(
            self, args=args, namespace=namespace)
        # handle any args that have is_write_out_config_file_arg set to true
        # check if the user specified this arg on the commandline
        if not ignore_write_args:
            output_file_paths = [getattr(namespace, a.dest, None) for a in self._actions
                                 if getattr(a, 'is_write_out_config_file_arg', False)]
            output_file_paths = [a for a in output_file_paths if a is not None]
            for a in output_file_paths:
                import pathlib
                pathlib.Path(a).parent.mkdir(exist_ok=True, parents=True)
            self.write_config_file(namespace, output_file_paths, exit_after=True)
        return namespace, unknown_args

    # def add_argument(self, *args, **kwargs):
    #     """
    #     Additional support for is_print_out_config_arg
    #     """
    #     ass


# # Unfortunately we have to monkey patch this in
# def add_argument(self, *args, **kwargs):
#     """
#     This method supports the same args as ArgumentParser.add_argument(..)
#     as well as the additional args below.

#     Arguments:
#         env_var: If set, the value of this environment variable will override
#             any config file or default values for this arg (but can itself
#             be overridden on the commandline). Also, if auto_env_var_prefix is
#             set in the constructor, this env var name will be used instead of
#             the automatic name.
#         is_config_file_arg: If True, this arg is treated as a config file path
#             This provides an alternative way to specify config files in place of
#             the ArgumentParser(fromfile_prefix_chars=..) mechanism.
#             Default: False
#         is_write_out_config_file_arg: If True, this arg will be treated as a
#             config file path, and, when it is specified, will cause
#             configargparse to write all current commandline args to this file
#             as config options and then exit.
#             Default: False
#     """

#     env_var = kwargs.pop("env_var", None)

#     is_config_file_arg = kwargs.pop(
#         "is_config_file_arg", None) or kwargs.pop(
#         "is_config_file", None)  # for backward compat.

#     is_write_out_config_file_arg = kwargs.pop(
#         "is_write_out_config_file_arg", None)

#     action = self.original_add_argument_method(*args, **kwargs)

#     action.is_positional_arg = not action.option_strings
#     action.env_var = env_var
#     action.is_config_file_arg = is_config_file_arg
#     action.is_write_out_config_file_arg = is_write_out_config_file_arg

#     if action.is_positional_arg and env_var:
#         raise ValueError("env_var can't be set for a positional arg.")
#     if action.is_config_file_arg and not isinstance(action, argparse._StoreAction):
#         raise ValueError("arg with is_config_file_arg=True must have "
#                          "action='store'")
#     if action.is_write_out_config_file_arg:
#         error_prefix = "arg with is_write_out_config_file_arg=True "
#         if not isinstance(action, argparse._StoreAction):
#             raise ValueError(error_prefix + "must have action='store'")
#         if is_config_file_arg:
#                 raise ValueError(error_prefix + "can't also have "
#                                                 "is_config_file_arg=True")

#     return action
