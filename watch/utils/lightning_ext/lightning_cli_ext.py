"""
This module is an exension of jsonargparse and lightning CLI that will respect
scriptconfig style arguments

References:
    https://github.com/Lightning-AI/lightning/issues/15038
"""
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.cli import LightningArgumentParser
import jsonargparse
# from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from jsonargparse.signatures import get_signature_parameters
from jsonargparse.signatures import get_doc_short_description
from jsonargparse.parameter_resolvers import ParamData
from typing import List, Set, Union, Optional, Tuple, Type, Any
# from typing import Any, Dict  # NOQA
# from pytorch_lightning.cli import _JSONARGPARSE_SIGNATURES_AVAILABLE  # NOQA
try:
    from pytorch_lightning.cli import ActionConfigFile
except Exception:
    from jsonargparse import ActionConfigFile  # NOQA
from pytorch_lightning.cli import Namespace
from jsonargparse.util import get_import_path, iter_to_set_str
# from typing import Callable, List, Type, Union
# from jsonargparse import class_from_function
# from pytorch_lightning.utilities.exceptions import MisconfigurationException
import inspect
from argparse import SUPPRESS
from jsonargparse.typing import is_final_class

from jsonargparse.actions import _ActionConfigLoad  # NOQA
from jsonargparse.optionals import get_doc_short_description  # NOQA
from jsonargparse.parameter_resolvers import (ParamData, get_parameter_origins, get_signature_parameters,)  # NOQA
from jsonargparse.typehints import ActionTypeHint, LazyInitBaseClass, is_optional  # NOQA
from jsonargparse.typing import is_final_class  # NOQA
from jsonargparse.util import LoggerProperty, get_import_path, is_subclass, iter_to_set_str  # NOQA
from jsonargparse.signatures import is_factory_class, is_pure_dataclass


kinds = inspect._ParameterKind
inspect_empty = inspect._empty

# class ActionConfigFile_Extension(ActionConfigFile):

#     @staticmethod
#     def apply_config(parser, cfg, dest, value) -> None:
#         import xdev
#         xdev.embed()
#         from jsonargparse.actions import _ActionSubCommands
#         from jsonargparse.actions import previous_config_context
#         from jsonargparse.actions import get_config_read_mode
#         from jsonargparse.actions import Path
#         from jsonargparse.actions import load_value
#         from jsonargparse.actions import get_loader_exceptions
#         from jsonargparse.link_arguments import skip_apply_links

#         value

#         with _ActionSubCommands.not_single_subcommand(), previous_config_context(cfg), skip_apply_links():
#             kwargs = {'env': False, 'defaults': False, '_skip_check': True, '_fail_no_subcommand': False}
#             try:
#                 cfg_path: Optional[Path] = Path(value, mode=get_config_read_mode())
#             except TypeError as ex_path:
#                 try:
#                     if isinstance(load_value(value), str):
#                         raise ex_path
#                     cfg_path = None
#                     cfg_file = parser.parse_string(value, **kwargs)
#                 except (TypeError,) + get_loader_exceptions() as ex_str:
#                     raise TypeError(f'Parser key "{dest}": {ex_str}') from ex_str
#             else:
#                 cfg_file = parser.parse_path(value, **kwargs)
#             cfg_merged = parser.merge_config(cfg_file, cfg)
#             cfg.__dict__.update(cfg_merged.__dict__)
#             if cfg.get(dest) is None:
#                 cfg[dest] = []
#             cfg[dest].append(cfg_path)


class LightningArgumentParser_Extension(LightningArgumentParser):
    """


    Refactor references:
        ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/pytorch_lightning/cli.py
        ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/jsonargparse/core.py
        ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/jsonargparse/signatures.py

    """

    """
    Keep in sync with ~/code/watch/watch/utils/lightning_ext/lightning_cli_ext.py

    See if we can do something to land this functionality upstream
    """

    def add_subclass_arguments(
        self,
        baseclass: Union[Type, Tuple[Type, ...]],
        nested_key: str,
        as_group: bool = True,
        skip: Optional[Set[str]] = None,
        instantiate: bool = True,
        required: bool = False,
        metavar: str = 'CONFIG | CLASS_PATH_OR_NAME | .INIT_ARG_NAME VALUE',
        help: str = 'One or more arguments specifying "class_path" and "init_args" for any subclass of %(baseclass_name)s.',
        **kwargs
    ):
        """Adds arguments to allow specifying any subclass of the given base class.

        This adds an argument that requires a dictionary with a "class_path"
        entry which must be a import dot notation expression. Optionally any
        init arguments for the class can be given in the "init_args" entry.
        Since subclasses can have different init arguments, the help does not
        show the details of the arguments of the base class. Instead a help
        argument is added that will print the details for a given class path.

        Args:
            baseclass: Base class or classes to use to check subclasses.
            nested_key: Key for nested namespace.
            as_group: Whether arguments should be added to a new argument group.
            skip: Names of parameters that should be skipped.
            required: Whether the argument group is required.
            metavar: Variable string to show in the argument's help.
            help: Description of argument to show in the help.
            **kwargs: Additional parameters like in add_class_arguments.

        Raises:
            ValueError: When given an invalid base class.
        """
        if is_final_class(baseclass):
            raise ValueError("Not allowed for classes that are final.")
        if type(baseclass) is not tuple:
            baseclass = (baseclass,)  # type: ignore
        if not all(inspect.isclass(c) for c in baseclass):
            raise ValueError('Expected "baseclass" argument to be a class or a tuple of classes.')

        # print(f'Parse add_subclass_arguments: function_or_class={baseclass}')
        doc_group = get_doc_short_description(baseclass[0], logger=self.logger)
        group = self._create_group_if_requested(
            baseclass,
            nested_key,
            as_group,
            doc_group,
            config_load=False,
            required=required,
            instantiate=False,
        )

        added_args: List[str] = []
        if skip is not None:
            skip = {nested_key + '.init_args.' + s for s in skip}
        param = ParamData(name=nested_key, annotation=Union[baseclass], component=baseclass)
        str_baseclass = iter_to_set_str(get_import_path(x) for x in baseclass)
        kwargs.update({
            'metavar': metavar,
            'help': (help % {'baseclass_name': str_baseclass}),
        })
        if 'default' not in kwargs:
            kwargs['default'] = SUPPRESS
        self._add_signature_parameter(
            group,
            None,
            param,
            added_args,
            skip,
            sub_configs=True,
            instantiate=instantiate,
            **kwargs
        )

    def _add_signature_arguments(
        self,
        function_or_class,
        method_name,
        nested_key,
        as_group: bool = True,
        as_positional: bool = False,
        skip=None,
        fail_untyped: bool = True,
        sub_configs: bool = False,
        instantiate: bool = True,
        linked_targets=None
    ) -> list[str]:
        """Adds arguments from parameters of objects based on signatures and docstrings.

        Args:
            function_or_class: Object from which to add arguments.
            method_name: Class method from which to add arguments.
            nested_key: Key for nested namespace.
            as_group: Whether arguments should be added to a new argument group.
            as_positional: Whether to add required parameters as positional arguments.
            skip: Names of parameters that should be skipped.
            fail_untyped: Whether to raise exception if a required parameter does not have a type.
            sub_configs: Whether subclass type hints should be loadable from inner config file.
            instantiate: Whether the class group should be instantiated by :code:`instantiate_classes`.

        Returns:
            The list of arguments added.

        Raises:
            ValueError: When there are required parameters without at least one valid type.
        """
        ## Create group if requested ##
        doc_group = get_doc_short_description(function_or_class, method_name, self.logger)
        component = getattr(function_or_class, method_name) if method_name else function_or_class
        group = self._create_group_if_requested(component, nested_key, as_group, doc_group, instantiate=instantiate)

        params = get_signature_parameters(function_or_class, method_name, logger=self.logger)

        if hasattr(function_or_class, '__scriptconfig__'):
            # print(f'Parse scriptconfig params for: function_or_class={function_or_class}')
            # Specify our own set of explicit parameters here
            # pretend like things in scriptconfig are from the signature
            import inspect
            # Hack to insert our method for explicit parameterization
            config_cls = function_or_class.__scriptconfig__
            if hasattr(config_cls, '__default__'):
                default = config_cls.__default__
            else:
                default = config_cls.default

            for key, value in default.items():
                # TODO can we make this compatability better?
                # Can we actually use the scriptconfig argparsing action?
                type = value.parsekw['type']
                if type is None or not isinstance(type, type):
                    annotation = inspect._empty
                else:
                    annotation = type
                param = ParamData(
                    name=key,
                    annotation=annotation,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=value.value,
                    doc=value.parsekw['help'],
                    component=function_or_class.__init__,
                    parent=function_or_class,
                )
                param._scfg_value = value

                # print(f'add scriptconfig {key=}')
                params.append(param)
        else:
            # print(f'Parse NON-scriptconfig params for: function_or_class={function_or_class}')
            ...

        ## Add parameter arguments ##
        added_args = []
        for param in params:
            self._add_signature_parameter(
                group,
                nested_key,
                param,
                added_args,
                skip,
                fail_untyped=fail_untyped,
                sub_configs=sub_configs,
                linked_targets=linked_targets,
                as_positional=as_positional,
            )
        # import ubelt as ub
        # print('added_args = {}'.format(ub.repr2(added_args, nl=1)))
        return added_args

    def _add_signature_parameter(
        self,
        group,
        nested_key: Optional[str],
        param,
        added_args: List[str],
        skip: Optional[Set[str]] = None,
        fail_untyped: bool = True,
        as_positional: bool = False,
        sub_configs: bool = False,
        instantiate: bool = True,
        linked_targets: Optional[Set[str]] = None,
        default: Any = inspect_empty,
        **kwargs
    ):
        name = param.name
        kind = param.kind
        annotation = param.annotation
        if default == inspect_empty:
            default = param.default
        is_required = default == inspect_empty
        src = get_parameter_origins(param.component, param.parent)
        skip_message = f'Skipping parameter "{name}" from "{src}" because of: '
        if not fail_untyped and annotation == inspect_empty:
            annotation = Any
            default = None if is_required else default
            is_required = False
        if is_required and linked_targets is not None and name in linked_targets:
            default = None
            is_required = False
        if kind in {kinds.VAR_POSITIONAL, kinds.VAR_KEYWORD} or \
           (not is_required and name[0] == '_') or \
           (annotation == inspect_empty and not is_required and default is None):
            return
        elif skip and name in skip:
            self.logger.debug(skip_message + 'Parameter requested to be skipped.')
            return
        if is_factory_class(default):
            default = param.parent.__dataclass_fields__[name].default_factory()
        if annotation == inspect_empty and not is_required:
            annotation = type(default)
        if 'help' not in kwargs:
            kwargs['help'] = param.doc
        if not is_required:
            kwargs['default'] = default
            if default is None and not is_optional(annotation, object):
                annotation = Optional[annotation]
        elif not as_positional:
            kwargs['required'] = True
        is_subclass_typehint = False
        is_final_class_typehint = is_final_class(annotation)
        dest = (nested_key + '.' if nested_key else '') + name
        args = [dest if is_required and as_positional else '--' + dest]
        if param.origin:
            group_name = '; '.join(str(o) for o in param.origin)
            if group_name in group.parser.groups:
                group = group.parser.groups[group_name]
            else:
                group = group.parser.add_argument_group(
                    f'Conditional arguments [origins: {group_name}]',
                    name=group_name,
                )
        if annotation in {str, int, float, bool} or \
           is_subclass(annotation, (str, int, float)) or \
           is_final_class_typehint or \
           is_pure_dataclass(annotation):
            kwargs['type'] = annotation
        elif annotation != inspect_empty:
            try:
                is_subclass_typehint = ActionTypeHint.is_subclass_typehint(annotation, all_subtypes=False)
                kwargs['type'] = annotation
                sub_add_kwargs: dict = {'fail_untyped': fail_untyped, 'sub_configs': sub_configs}
                if is_subclass_typehint:
                    prefix = name + '.init_args.'
                    subclass_skip = {s[len(prefix):] for s in skip or [] if s.startswith(prefix)}
                    sub_add_kwargs['skip'] = subclass_skip
                args = ActionTypeHint.prepare_add_argument(
                    args=args,
                    kwargs=kwargs,
                    enable_path=is_subclass_typehint and sub_configs,
                    container=group,
                    logger=self.logger,
                    sub_add_kwargs=sub_add_kwargs,
                )
            except ValueError as ex:
                self.logger.debug(skip_message + str(ex))
        if 'type' in kwargs or 'action' in kwargs:
            sub_add_kwargs = {
                'fail_untyped': fail_untyped,
                'sub_configs': sub_configs,
                'instantiate': instantiate,
            }
            if is_final_class_typehint:
                kwargs.update(sub_add_kwargs)

            if hasattr(param, '_scfg_value'):
                value = param._scfg_value
                _value = value

                def _resolve_alias(name, _value, fuzzy_hyphens):
                    if _value is None:
                        aliases = None
                        short_aliases = None
                    else:
                        aliases = _value.alias
                        short_aliases = _value.short_alias
                    if isinstance(aliases, str):
                        aliases = [aliases]
                    if isinstance(short_aliases, str):
                        short_aliases = [short_aliases]
                    long_names = [name] + list((aliases or []))
                    short_names = list(short_aliases or [])
                    if fuzzy_hyphens:
                        # Do we want to allow for people to use hyphens on the CLI?
                        # Maybe, we can make it optional.
                        unique_long_names = set(long_names)
                        modified_long_names = {n.replace('_', '-') for n in unique_long_names}
                        extra_long_names = modified_long_names - unique_long_names
                        long_names += sorted(extra_long_names)
                    nest_prefix = (nested_key + '.' if nested_key else '')
                    short_option_strings = ['-' + nest_prefix + n for n in short_names]
                    long_option_strings = ['--' + nest_prefix +  n for n in long_names]
                    option_strings = short_option_strings + long_option_strings
                    return option_strings

                args = _resolve_alias(name, _value, fuzzy_hyphens=0)
                # print(f'long_option_strings={long_option_strings}')
                # print(f'short_option_strings={short_option_strings}')

            action = group.add_argument(*args, **kwargs)
            action.sub_add_kwargs = sub_add_kwargs
            if is_subclass_typehint and len(subclass_skip) > 0:
                action.sub_add_kwargs['skip'] = subclass_skip
            added_args.append(dest)
        elif is_required and fail_untyped:
            raise ValueError(f'Required parameter without a type for "{src}" parameter "{name}".')


# Monkey patch jsonargparse so its subcommands use our extended functionality
jsonargparse.ArgumentParser = LightningArgumentParser_Extension
jsonargparse.core.ArgumentParser = LightningArgumentParser_Extension


# Should try to patch into upstream
class LightningCLI_Extension(LightningCLI):
    ...

    def init_parser(self, **kwargs):
        # Hack in our modified parser
        DEBUG = 0
        if DEBUG:
            kwargs['error_handler'] = None
        import pytorch_lightning as pl
        kwargs.setdefault("dump_header", [f"pytorch_lightning=={pl.__version__}"])
        parser = LightningArgumentParser_Extension(**kwargs)
        parser.add_argument(
            "-c", "--config", action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format."
        )
        return parser

    def parse_arguments(self, parser: LightningArgumentParser, args) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        import sys
        if args is not None and len(sys.argv) > 1:
            # Please let us shoot ourselves in the foot.
            import warnings
            warnings.warn(
                "LightningCLI's args parameter is intended to run from within Python like if it were from the command "
                "line. To prevent mistakes it is not allowed to provide both args and command line arguments, got: "
                f"sys.argv[1:]={sys.argv[1:]}, args={args}."
            )
        if isinstance(args, (dict, Namespace)):
            self.config = parser.parse_object(args)
        else:
            self.config = parser.parse_args(args)
