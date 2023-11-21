"""
Patches for jsonargparse version >= 4.21.0

Refactor references:
    ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/pytorch_lightning/cli.py
    ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/jsonargparse/core.py
    ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/jsonargparse/signatures.py

Keep in sync with ~/code/watch/geowatch/utils/lightning_ext/lightning_cli_ext.py
See if we can do something to land this functionality upstream
"""
from argparse import Action as ArgparseAction
from argparse import SUPPRESS
from jsonargparse.actions import _ActionConfigLoad
from jsonargparse.namespace import Namespace
from jsonargparse.optionals import get_doc_short_description
from jsonargparse.parameter_resolvers import ParamData
from jsonargparse.parameter_resolvers import get_parameter_origins
from jsonargparse.parameter_resolvers import get_signature_parameters
from jsonargparse.signatures import is_factory_class
from jsonargparse.type_checking import ArgumentParser
from jsonargparse.typehints import ActionTypeHint, is_optional
from typing import Callable
from typing import Dict
from typing import List, Set, Union, Optional, Tuple, Type, Any
import inspect

from jsonargparse._common import is_dataclass_like
from jsonargparse._common import is_subclass
from jsonargparse.signatures import iter_to_set_str
from jsonargparse.typing import register_pydantic_type
from jsonargparse.signatures import get_import_path

kinds = inspect._ParameterKind
inspect_empty = inspect._empty


def extract_scriptconfig_params(function_or_class):
    scriptconfig_params = []
    if hasattr(function_or_class, '__scriptconfig__'):
        # print(f'Parse scriptconfig params for: function_or_class={function_or_class}')
        # Specify our own set of explicit parameters here
        # pretend like things in scriptconfig are from the signature
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
            scriptconfig_params.append(param)
    else:
        # print(f'Parse NON-scriptconfig params for: function_or_class={function_or_class}')
        ...
    return scriptconfig_params


def extract_scriptconfig_param_args(param, nested_key):
    """
    Get option strings that reflect scriptconfig aliases
    """
    name = param.name
    _value = param._scfg_value
    fuzzy_hyphens = 0

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
    args = option_strings
    # print(f'long_option_strings={long_option_strings}')
    # print(f'short_option_strings={short_option_strings}')
    return args


class ArgumentParserPatches:
    """
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
        if is_dataclass_like(baseclass):
            raise ValueError("Not allowed for dataclass-like classes.")
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
        nested_key: Optional[str],
        as_group: bool = True,
        as_positional: bool = False,
        skip: Optional[Set[Union[str, int]]] = None,
        fail_untyped: bool = True,
        sub_configs: bool = False,
        instantiate: bool = True,
        linked_targets: Optional[Set[str]] = None,
    ) -> List[str]:
        """Adds arguments from parameters of objects based on signatures and docstrings.

        Args:
            function_or_class: Object from which to add arguments.
            method_name: Class method from which to add arguments.
            nested_key: Key for nested namespace.
            as_group: Whether arguments should be added to a new argument group.
            as_positional: Whether to add required parameters as positional arguments.
            skip: Names of parameters or number of positionals that should be skipped.
            fail_untyped: Whether to raise exception if a required parameter does not have a type.
            sub_configs: Whether subclass type hints should be loadable from inner config file.
            instantiate: Whether the class group should be instantiated by :code:`instantiate_classes`.

        Returns:
            The list of arguments added.

        Raises:
            ValueError: When there are required parameters without at least one valid type.
        """
        params = get_signature_parameters(function_or_class, method_name, logger=self.logger)

        params += extract_scriptconfig_params(function_or_class)

        skip_positionals = [s for s in (skip or []) if isinstance(s, int)]
        if skip_positionals:
            if len(skip_positionals) > 1 or any(p <= 0 for p in skip_positionals):
                raise ValueError(f'Unexpected number of positionals to skip: {skip_positionals}')
            names = {p.name for p in params[:skip_positionals[0]]}
            params = params[skip_positionals[0]:]
            self.logger.debug(f'Skipping parameters {names} because {skip_positionals[0]} positionals requested to be skipped.')

        ## Create group if requested ##
        doc_group = get_doc_short_description(function_or_class, method_name, logger=self.logger)
        component = getattr(function_or_class, method_name) if method_name else function_or_class
        group = self._create_group_if_requested(
            component,
            nested_key,
            as_group,
            doc_group,
            config_load=len(params) > 0,
            instantiate=instantiate,
        )

        ## Add parameter arguments ##
        added_args: List[str] = []
        for param in params:
            self._add_signature_parameter(
                group,
                nested_key,
                param,
                added_args,
                skip={s for s in (skip or []) if isinstance(s, str)},
                fail_untyped=fail_untyped,
                sub_configs=sub_configs,
                linked_targets=linked_targets,
                as_positional=as_positional,
            )
        # import ubelt as ub
        # print('added_args = {}'.format(ub.urepr(added_args, nl=1)))
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
        is_dataclass_like_typehint = is_dataclass_like(annotation)
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
           is_dataclass_like_typehint:
            kwargs['type'] = annotation
            register_pydantic_type(annotation)
        elif annotation != inspect_empty:
            try:
                is_subclass_typehint = ActionTypeHint.is_subclass_typehint(annotation, all_subtypes=False)
                kwargs['type'] = annotation
                sub_add_kwargs: dict = {'fail_untyped': fail_untyped, 'sub_configs': sub_configs}
                if is_subclass_typehint:
                    prefix = name + '.init_args.'
                    subclass_skip = {s[len(prefix):] for s in skip or [] if s.startswith(prefix)}
                    sub_add_kwargs['skip'] = subclass_skip
                else:
                    register_pydantic_type(annotation)
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
            if is_dataclass_like_typehint:
                kwargs.update(sub_add_kwargs)

            if hasattr(param, '_scfg_value'):
                args = extract_scriptconfig_param_args(param, nested_key)

            action = group.add_argument(*args, **kwargs)
            action.sub_add_kwargs = sub_add_kwargs
            if is_subclass_typehint and len(subclass_skip) > 0:
                action.sub_add_kwargs['skip'] = subclass_skip
            added_args.append(dest)
        elif is_required and fail_untyped:
            msg = f'With fail_untyped=True, all mandatory parameters must have a supported type. Parameter "{name}" from "{src}" '
            if isinstance(annotation, str):
                msg += 'specifies the type as a string. Types as a string and `from __future__ import annotations` is currently not supported.'
            else:
                msg += 'does not specify a type.'
            raise ValueError(msg)

    def _apply_actions(
        self,
        cfg: Union[Namespace, Dict[str, Any]],
        parent_key: str = '',
        prev_cfg: Optional[Namespace] = None,
        skip_fn: Optional[Callable[[Any], bool]] = None,
    ) -> Namespace:
        """Runs _check_value_key on actions present in config."""
        from jsonargparse.core import ActionJsonnet
        from jsonargparse.core import ActionJsonnetExtVars
        from jsonargparse.core import _ActionSubCommands
        from jsonargparse.core import parser_context

        if isinstance(cfg, dict):
            cfg = Namespace(cfg)
        if parent_key:
            cfg_branch = cfg
            cfg = Namespace()
            cfg[parent_key] = cfg_branch
            keys = [parent_key + '.' + k for k in cfg_branch.__dict__.keys()]
        else:
            keys = list(cfg.__dict__.keys())

        if prev_cfg:
            prev_cfg = prev_cfg.clone()
        else:
            prev_cfg = Namespace()

        config_keys: Set[str] = set()
        num = 0
        while num < len(keys):
            key = keys[num]
            exclude = _ActionConfigLoad if key in config_keys else None
            action, subcommand = _find_action_and_subcommand(self, key, exclude=exclude)

            if isinstance(action, ActionJsonnet):
                ext_vars_key = action._ext_vars
                if ext_vars_key and ext_vars_key not in keys[:num]:
                    keys = keys[:num] + [ext_vars_key] + [k for k in keys[num:] if k != ext_vars_key]
                    continue

            num += 1

            if action is None or isinstance(action, _ActionSubCommands):
                value = cfg[key]
                if isinstance(value, dict):
                    value = Namespace(value)
                if isinstance(value, Namespace):
                    new_keys = value.__dict__.keys()
                    keys += [key + '.' + k for k in new_keys if key + '.' + k not in keys]
                cfg[key] = value
                continue

            action_dest = action.dest if subcommand is None else subcommand + '.' + action.dest
            try:
                value = cfg[action_dest]
            except KeyError:
                # If the main key isn't in the config, check if it exists
                # under an alias.
                found = None
                for alias in _action_aliases(action):
                    if alias in cfg:
                        value = cfg[alias]
                        found = True
                        break
                if not found:
                    raise
            if skip_fn and skip_fn(value):
                continue
            with parser_context(parent_parser=self, lenient_check=True):
                value = self._check_value_key(action, value, action_dest, prev_cfg)
            if isinstance(action, _ActionConfigLoad):
                config_keys.add(action_dest)
                keys.append(action_dest)
            elif isinstance(action, ActionJsonnetExtVars):
                prev_cfg[action_dest] = value
            cfg[action_dest] = value
        return cfg[parent_key] if parent_key else cfg


def _find_action_and_subcommand(
    parser: 'ArgumentParser',
    dest: str,
    exclude: Optional[Union[Type[ArgparseAction], Tuple[Type[ArgparseAction], ...]]] = None,
) -> Tuple[Optional[ArgparseAction], Optional[str]]:
    """Finds an action in a parser given its destination key.

    Args:
        parser: A parser where to search.
        dest: The destination key to search with.

    Returns:
        The action if found, otherwise None.
    """
    from jsonargparse.actions import filter_default_actions
    from jsonargparse.actions import _ActionSubCommands
    from jsonargparse.actions import split_key_root
    actions = filter_default_actions(parser._actions)
    if exclude is not None:
        actions = [a for a in actions if not isinstance(a, exclude)]
    fallback_action = None

    for action in actions:
        # _StoreAction seems to break the property
        # if dest in action.aliases:
        if dest in _action_aliases(action):
            if isinstance(action, _ActionConfigLoad):
                fallback_action = action
            else:
                return action, None
        elif isinstance(action, _ActionSubCommands):
            if dest in action._name_parser_map:
                return action, None
            elif split_key_root(dest)[0] in action._name_parser_map:
                subcommand, subdest = split_key_root(dest)
                subparser = action._name_parser_map[subcommand]
                subaction, subsubcommand = _find_action_and_subcommand(subparser, subdest, exclude=exclude)
                if subsubcommand is not None:
                    subcommand += '.' + subsubcommand
                return subaction, subcommand
    return fallback_action, None


def _action_aliases(self):
    if not hasattr(self, '_aliases'):
        options = {optstr.lstrip('-').replace('-', '_')
                   for optstr in self.option_strings}
        options = {opt for opt in options if len(opt) > 1}
        self._aliases =  {self.dest} | options
    return self._aliases
