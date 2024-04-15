"""
Argparse Extensions
"""
import argparse
import os
import sys
_FALSY = {'0', 'false', 'f', 'no', ''}
SCRIPTCONFIG_NORICH = os.environ.get('SCRIPTCONFIG_NORICH', '').lower() not in _FALSY


__docstubs__ = """
import argparse
_Base = argparse._StoreAction

_RawDescriptionHelpFormatter = argparse.RawDescriptionHelpFormatter
_ArgumentDefaultsHelpFormatter = argparse.ArgumentDefaultsHelpFormatter
"""

try:
    if SCRIPTCONFIG_NORICH:
        raise ImportError
    import rich_argparse
except ImportError:
    _RawDescriptionHelpFormatter = argparse.RawDescriptionHelpFormatter
    _ArgumentDefaultsHelpFormatter = argparse.ArgumentDefaultsHelpFormatter
else:
    _RawDescriptionHelpFormatter = rich_argparse.RawDescriptionRichHelpFormatter
    _ArgumentDefaultsHelpFormatter = rich_argparse.ArgumentDefaultsRichHelpFormatter


# Check if we are on 3.11 with a patch version higher than 3.11.9
# Or if we are higher than 3.12.3
# https://github.com/python/cpython/pull/115674
HAS_ARGPARSE_GH_114180 = (
    (sys.version_info[0:2] == (3, 11) and sys.version_info[2] >= 9) or
    (sys.version_info[0:3] >= (3, 12, 3))
)

# Inherit from StoreAction to make configargparse happy.  Hopefully python
# doesn't change the behavior of this private class.
# If we ditch support for configargparse in the future, then we can more
# reasonably just inherit from Action
_Base = argparse._StoreAction
# _Base = argparse.Action


class BooleanFlagOrKeyValAction(_Base):
    """
    An action that allows you to specify a boolean via a flag as per usual
    or a key/value pair.

    This helps allow for a flexible specification of boolean values:

        --flag        > {'flag': True}
        --flag=1      > {'flag': True}
        --flag True   > {'flag': True}
        --flag True   > {'flag': True}
        --flag False  > {'flag': False}
        --flag 0      > {'flag': False}
        --no-flag     > {'flag': False}
        --no-flag=0   > {'flag': True}
        --no-flag=1   > {'flag': False}
    """
    def __init__(self, option_strings, dest, default=None, required=False,
                 help=None):

        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)
            if option_string.startswith('--'):
                option_string = '--no-' + option_string[2:]
                _option_strings.append(option_string)
        if help is not None and default is not None and default is not argparse.SUPPRESS:
            help += " (default: %(default)s)"

        actionkw = dict(
            option_strings=_option_strings, dest=dest, default=default,
            type=None, choices=None, required=required, help=help,
            metavar=None)
        # Either the zero arg flag form or the 1 arg key/value form.
        actionkw['nargs'] = '?'

        # Hack because of the Store Base for configargparse support
        argparse.Action.__init__(self, **actionkw)
        # super().__init__(**actionkw)

    def format_usage(self):
        # I thought this was used in formatting the help, but it seems like
        # we dont have much control over that here.
        if self.default is False:
            # If the default is false, don't show the negative variants
            _option_strings = []
            for option_string in self.option_strings:
                if not option_string.startswith('--no'):
                    _option_strings.append(option_string)
        else:
            _option_strings = self.option_strings
        return ' | '.join(_option_strings)

    def _mark_parsed_argument(action, parser):
        if not hasattr(parser, '_explicitly_given'):
            # We might be given a subparser / parent parser
            # and not the original one we created.
            parser._explicitly_given = set()
        parser._explicitly_given.add(action.dest)

    def __call__(action, parser, namespace, values, option_string=None):
        if option_string in action.option_strings:
            # Was the positive or negated key given?
            key_default = not option_string.startswith('--no-')
        # Was there a value or was the flag specified by itself?
        if values is None:
            value = key_default
        else:
            # Allow for non-boolean values (i.e. auto) to be passed
            from scriptconfig import smartcast as smartcast_mod
            value = smartcast_mod.smartcast(values)
            # value = smartcast_mod._smartcast_bool(values)
            if not key_default:
                value = not value
        setattr(namespace, action.dest, value)
        action._mark_parsed_argument(parser)


class CounterOrKeyValAction(BooleanFlagOrKeyValAction):
    """
    Extends :BooleanFlagOrKeyValAction: and will increment the value
    based on the number of times the flag is specified.

    FIXME:
        Can we get -ffff to work right?

    Example:
        >>> from scriptconfig.argparse_ext import *  # NOQA
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('-f', '--flag', action=CounterOrKeyValAction)
        >>> print(parser.format_usage())
        >>> print(parser.format_help())
        >>> import shlex
        >>> # Map the CLI arg string to what value we would expect to get
        >>> variants = {
        >>>     # Case1: you either specify the flag, or you don't
        >>>     '': None,
        >>>     '--flag': True,
        >>>     '--no-flag': False,
        >>>     # Case1: You specify the flag as a key/value pair
        >>>     '--flag=0': False,
        >>>     '--flag=1': True,
        >>>     '--flag True': True,
        >>>     '--flag False': False,
        >>>     # Case1: You specify the negated flag as a key/value pair
        >>>     # (you probably shouldn't do this)
        >>>     '--no-flag 0': True,
        >>>     '--no-flag 1': False,
        >>>     '--no-flag=True': False,
        >>>     '--no-flag=False': True,
        >>>     # Multiple flag specification cases
        >>>     '--flag --flag --flag': 3,
        >>>     # An explicit set overwrites previous increments
        >>>     '--flag --flag --flag --flag=0': 0,
        >>>     # An increments modify previous explicit settings
        >>>     '--flag=3 --flag --flag --flag': 6,
        >>> }
        >>> for args, want in variants.items():
        >>>     args = shlex.split(args)
        >>>     ns = parser.parse_known_args(args=args)[0].__dict__
        >>>     print(f'args={args} -> {ns}')
        >>>     assert ns['flag'] == want
    """
    def __call__(action, parser, namespace, values, option_string=None):
        if option_string in action.option_strings:
            # Was the positive or negated key given?
            key_default = not option_string.startswith('--no-')

        # Was there a value or was the flag specified by itself?
        if values is None:
            # For the no k/v case, allow incrementing of the value
            prev_value = getattr(namespace, action.dest)
            if prev_value is None:
                prev_value = 0
            value = prev_value + key_default
        else:
            # Allow for non-boolean values (i.e. auto) to be passed
            from scriptconfig import smartcast as smartcast_mod
            value = smartcast_mod.smartcast(values)
            # value = smartcast_mod._smartcast_bool(values)
            if not key_default:
                value = not value

        setattr(namespace, action.dest, value)
        action._mark_parsed_argument(parser)


class RawDescriptionDefaultsHelpFormatter(
        _RawDescriptionHelpFormatter,
        _ArgumentDefaultsHelpFormatter):

    group_name_formatter = str  # revert rich-argparse title change

    def _concise_option_strings(self, action):
        # When working with fuzzy hyphens only show one variant of each
        # possibility.
        display_option_strings = []
        _seen = set()
        for s in action.option_strings:
            _norm = s.replace('_', '-')
            if _norm not in _seen:
                _seen.add(_norm)
                display_option_strings.append(s)
        return display_option_strings

    def _format_action_invocation(self, action):
        """
        Custom mixin to reduce clutter from accepting fuzzy hyphens
        """
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar

        else:
            parts = []

            SCFG_MODIFICATIONS = True
            if SCFG_MODIFICATIONS:
                display_option_strings = self._concise_option_strings(action)
            else:
                display_option_strings = action.option_strings

            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(display_option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in display_option_strings:
                    if SCFG_MODIFICATIONS:
                        if option_string.startswith('--no-'):
                            if isinstance(action.default, int) and action.default == 0:
                                # Dont bother telling the user they can turn
                                # something off when that is the default.
                                continue
                            parts.append('%s' % (option_string,))
                        else:
                            parts.append('%s %s' % (option_string, args_string))
                    else:
                        parts.append('%s %s' % (option_string, args_string))
            return ', '.join(parts)

    def _rich_format_action_invocation(self, action):
        """
        Mirrors _format_action_invocation but for rich-argparse
        """
        from rich.text import Text

        if not action.option_strings:
            return Text().append(self._format_action_invocation(action), style="argparse.args")
        else:
            parts = []
            SCFG_MODIFICATIONS = True
            if SCFG_MODIFICATIONS:
                display_option_strings = self._concise_option_strings(action)
            else:
                display_option_strings = action.option_strings

            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend([Text(o, 'argparse.args') for o in display_option_strings])

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in display_option_strings:
                    if option_string.startswith('--no-'):
                        if isinstance(action.default, int) and action.default == 0:
                            # Dont bother telling the user they can turn
                            # something off when that is the default.
                            continue
                        part = Text(option_string, 'argparse.args')
                    else:
                        part = Text(" ").join([Text(option_string, 'argparse.args'), Text(args_string, 'argparse.metavar')])
                    parts.append(part)
            return Text(", ").join(parts)


class CompatArgumentParser(argparse.ArgumentParser):
    """
    A modified version of the standard library ArgumentParser with back-ported
    features needed by scriptconfig for compatability across different Python
    versions. Namely, this ensures the ``exit_on_error`` property exists for
    Python 3.6 - 3.8
    """

    def __init__(self, *args, **kwargs):
        self.exit_on_error = kwargs.pop('exit_on_error', True)
        super().__init__(*args, **kwargs)

    def parse_known_args(self, args=None, namespace=None):
        """
        This is the Python 3.10 implementation of this function.
        We define this for Python 3.6-3.8 compatibility where the exit_on_error
        flag does not exist.
        """
        # This is the version from Python 3.10
        from argparse import _sys, Namespace, SUPPRESS, ArgumentError
        from argparse import _UNRECOGNIZED_ARGS_ATTR
        import os
        if args is None:
            # args default to the system args
            args = _sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)
            # Allow Paths objects
            args = [os.fspath(a) if isinstance(a, os.PathLike) else a for a in args]

        # default Namespace built from parser defaults
        if namespace is None:
            namespace = Namespace()

        # add any action defaults that aren't present
        for action in self._actions:
            if action.dest is not SUPPRESS:
                if not hasattr(namespace, action.dest):
                    if action.default is not SUPPRESS:
                        setattr(namespace, action.dest, action.default)

        # add any parser defaults that aren't present
        for dest in self._defaults:
            if not hasattr(namespace, dest):
                setattr(namespace, dest, self._defaults[dest])

        # parse the arguments and exit if there are any errors
        if self.exit_on_error:
            try:
                namespace, args = self._parse_known_args(args, namespace)
            except ArgumentError:
                err = _sys.exc_info()[1]
                self.error(str(err))
        else:
            namespace, args = self._parse_known_args(args, namespace)

        if hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
            args.extend(getattr(namespace, _UNRECOGNIZED_ARGS_ATTR))
            delattr(namespace, _UNRECOGNIZED_ARGS_ATTR)
        return namespace, args


class ExtendedArgumentParser_PRE_GH_114180(CompatArgumentParser):
    """
    Extends the compatable argument parser to add minor new features.
    Namely: allowing options in argv to interchangably use "_" or "-".

    This is based on a 2018 version (~cpython 3.7.2) of argparse.
    E.g. https://github.com/python/cpython/blob/v3.7.2/Lib/argparse.py

    References:
        .. [SO53527387] https://stackoverflow.com/questions/53527387/make-argparse-treat-dashes-and-underscore-identically
    """

    def _parse_optional(self, arg_string):
        """
        Allow "_" or "-" on the CLI.

        """
        from gettext import gettext as gettext_fn

        # if it's an empty string, it was meant to be a positional
        if not arg_string:
            return None

        # if it doesn't start with a prefix, it was meant to be positional
        if not arg_string[0] in self.prefix_chars:
            return None

        # if it's just a single character, it was meant to be positional
        if len(arg_string) == 1:
            return None

        option_tuples = self._get_option_tuples(arg_string)

        # if multiple actions match, the option string was ambiguous
        if len(option_tuples) > 1:
            options = ', '.join([option_string
                                 for action, option_string, explicit_arg in option_tuples])
            args = {'option': arg_string, 'matches': options}
            msg = gettext_fn('ambiguous option: %(option)s could match %(matches)s')
            self.error(msg % args)

        # if exactly one action matched, this segmentation is good,
        # so return the parsed action
        elif len(option_tuples) == 1:
            option_tuple, = option_tuples
            return option_tuple

        # if it was not found as an option, but it looks like a negative
        # number, it was meant to be positional
        # unless there are negative-number-like options
        if self._negative_number_matcher.match(arg_string):
            if not self._has_negative_number_optionals:
                return None

        # if it contains a space, it was meant to be a positional
        if ' ' in arg_string:
            return None

        # it was meant to be an optional but there is no such option
        # in this parser (though it might be a valid option in a subparser)
        return None, arg_string, None

    def _get_option_tuples(self, option_string):
        """
        Helper to allow "_" or "-" on the CLI.
        """
        result = []

        if '=' in option_string:
            option_prefix, explicit_arg = option_string.split('=', 1)
        else:
            option_prefix = option_string
            explicit_arg = None
        if option_prefix in self._option_string_actions:
            action = self._option_string_actions[option_prefix]
            tup = action, option_prefix, explicit_arg
            result.append(tup)
        else:  # imperfect match
            chars = self.prefix_chars
            if option_string[0] in chars and option_string[1] not in chars:
                # short option: if single character, can be concatenated with arguments
                short_option_prefix = option_string[:2]
                short_explicit_arg = option_string[2:]
                if short_option_prefix in self._option_string_actions:
                    action = self._option_string_actions[short_option_prefix]
                    # FIXME: An update to CPython 3.11 added a new "sep"
                    # pararameter in the option tuple.
                    # Commit that broke us is here:
                    # https://github.com/python/cpython/commit/c02b7ae4dd367444aa6822d5fb73b61e8f5a4ff9
                    tup = action, short_option_prefix, short_explicit_arg
                    result.append(tup)

            underscored = {k.replace('-', '_'): k for k in self._option_string_actions}
            option_prefix = option_prefix.replace('-', '_')
            if option_prefix in underscored:
                action = self._option_string_actions[underscored[option_prefix]]
                tup = action, underscored[option_prefix], explicit_arg
                result.append(tup)
            elif self.allow_abbrev:
                for option_string in underscored:
                    if option_string.startswith(option_prefix):
                        action = self._option_string_actions[underscored[option_string]]
                        tup = action, underscored[option_string], explicit_arg
                        result.append(tup)

        # return the collected option tuples
        return result


class ExtendedArgumentParser_POST_GH_114180(CompatArgumentParser):
    """
    Extends the compatable argument parser to add minor new features.
    Namely: allowing options in argv to interchangably use "_" or "-".

    This is based on the CPython 3.12.3 versin of of argparse.
    https://github.com/python/cpython/blob/v3.7.2/Lib/argparse.py

    This is an alternate version of
    :class:`ExtendedArgumentParser_PRE_GH_114180` that works with changes
    introduced in
    https://github.com/python/cpython/commit/c02b7ae4dd367444aa6822d5fb73b61e8f5a4ff9
    """
    def _get_option_tuples(self, option_string):
        result = []

        # option strings starting with two prefix characters are only
        # split at the '='
        chars = self.prefix_chars
        if option_string[0] in chars and option_string[1] in chars:
            if self.allow_abbrev:
                option_prefix, sep, explicit_arg = option_string.partition('=')
                norm_option_prefix = option_prefix.replace('-', '_')
                if not sep:
                    sep = explicit_arg = None
                for option_string in self._option_string_actions:
                    norm_option_string = option_string.replace('-', '_')
                    # if option_string.startswith(option_prefix):
                    if norm_option_string.startswith(norm_option_prefix):
                        action = self._option_string_actions[option_string]
                        tup = action, option_string, sep, explicit_arg
                        result.append(tup)

        # single character options can be concatenated with their arguments
        # but multiple character options always have to have their argument
        # separate
        elif option_string[0] in chars and option_string[1] not in chars:
            option_prefix = option_string
            short_option_prefix = option_string[:2]
            short_explicit_arg = option_string[2:]

            for option_string in self._option_string_actions:
                if option_string == short_option_prefix:
                    action = self._option_string_actions[option_string]
                    tup = action, option_string, '', short_explicit_arg
                    result.append(tup)
                elif option_string.startswith(option_prefix):
                    action = self._option_string_actions[option_string]
                    tup = action, option_string, None, None
                    result.append(tup)

        # shouldn't ever get here
        else:
            self.error(('unexpected option string: %s') % option_string)

        # return the collected option tuples
        return result


if HAS_ARGPARSE_GH_114180:
    _ExtendedArgumentParserBase = ExtendedArgumentParser_POST_GH_114180
else:
    _ExtendedArgumentParserBase = ExtendedArgumentParser_PRE_GH_114180


class ExtendedArgumentParser(_ExtendedArgumentParserBase):
    """
    Extends the compatable argument parser to add minor new features.
    Namely: allowing options in argv to interchangably use "_" or "-".
    """


def patch_0_7_14():
    # Patch issue in scriptconfig without requiring a dependency update
    # This can be removed if we bump the min scriptconfig version to 0.7.14
    from scriptconfig import argparse_ext
    argparse_ext.CompatArgumentParser = ExtendedArgumentParser
    argparse_ext.ExtendedArgumentParser = ExtendedArgumentParser
