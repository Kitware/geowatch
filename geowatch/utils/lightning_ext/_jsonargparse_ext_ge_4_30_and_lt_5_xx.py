"""
Follows the patches submitted in https://github.com/omni-us/jsonargparse/pull/532

Refactor References:
    ~/code/jsonargparse/jsonargparse/_signatures.py
    ~/code/jsonargparse/jsonargparse/_parameter_resolvers.py

"""
import dataclasses
import inspect
from typing import Any, Callable, List, Optional, Set, Tuple, Type, Union
# from jsonargparse._common import LoggerProperty
from jsonargparse._parameter_resolvers import get_signature_parameters
from jsonargparse._parameter_resolvers import get_parameter_origins
from jsonargparse._parameter_resolvers import ParamData
from jsonargparse._optionals import get_doc_short_description
from jsonargparse._signatures import SignatureArguments
from argparse import ArgumentParser
from jsonargparse._common import LoggerProperty   # NOQA
from jsonargparse._common import get_class_instantiator   # NOQA
from jsonargparse._common import get_generic_origin   # NOQA
from jsonargparse._common import get_unaliased_type   # NOQA
from jsonargparse._common import is_dataclass_like   # NOQA
from jsonargparse._common import is_subclass   # NOQA
from jsonargparse._typehints import ActionTypeHint
from jsonargparse._typehints import is_optional
from jsonargparse.typing import register_pydantic_type

kinds = inspect._ParameterKind
inspect_empty = inspect._empty


@dataclasses.dataclass
class ParamData_Extension(ParamData):
    name: str
    annotation: Any
    default: Any = inspect._empty
    kind: Optional[inspect._ParameterKind] = None
    doc: Optional[str] = None
    component: Optional[Union[Callable, Type, Tuple]] = None
    parent: Optional[Union[Type, Tuple]] = None
    origin: Optional[Union[str, Tuple]] = None
    short_aliases: Optional[List[str]] = None
    long_aliases: Optional[List[str]] = None

    def _resolve_args_and_dest(self, is_required=False, as_positional=False, nested_key: Optional[str] = None):
        name = self.name
        dest = (nested_key + "." if nested_key else "") + name
        if is_required and as_positional:
            args = [dest]
        else:
            long_names = [name] + list((self.long_aliases or []))
            short_names = list(self.short_aliases or [])
            nest_prefix = nested_key + "." if nested_key else ""
            short_option_strings = ["-" + nest_prefix + n for n in short_names]
            long_option_strings = ["--" + nest_prefix + n for n in long_names]
            args = short_option_strings + long_option_strings
        return args, dest


class SignatureArguments_Extension(SignatureArguments):

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

        skip_positionals = [s for s in (skip or []) if isinstance(s, int)]
        if skip_positionals:
            if len(skip_positionals) > 1 or any(p <= 0 for p in skip_positionals):
                raise ValueError(f"Unexpected number of positionals to skip: {skip_positionals}")
            names = {p.name for p in params[: skip_positionals[0]]}
            params = params[skip_positionals[0] :]
            self.logger.debug(
                f"Skipping parameters {names} because {skip_positionals[0]} positionals requested to be skipped."
            )

        prefix = "--" + (nested_key + "." if nested_key else "")
        for param in params:
            if skip and param.name in skip:
                continue
            if prefix + param.name in self._option_string_actions:  # type: ignore[attr-defined]
                raise ValueError(
                    f"Unable to add parameter '{param.name}' from {function_or_class} because "
                    f"argument '{prefix + param.name}' already exists."
                )

        ## Create group if requested ##
        doc_group = get_doc_short_description(function_or_class, method_name, logger=self.logger)
        component = getattr(function_or_class, method_name) if method_name else function_or_class
        container = self._create_group_if_requested(
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
                container,
                nested_key,
                param,
                added_args,
                skip={s for s in (skip or []) if isinstance(s, str)},
                fail_untyped=fail_untyped,
                sub_configs=sub_configs,
                linked_targets=linked_targets,
                as_positional=as_positional,
            )

        if hasattr(function_or_class, "__scriptconfig__"):
            # Integrate with scriptconfig style classes.
            # When a function/class has a __scriptconfig__ object that means it
            # should accept any of the options defined in the config as keyword
            # arguments. We can utilize additional metadata stored in the
            # scriptconfig object to enrich our CLI.
            config_cls = function_or_class.__scriptconfig__
            self._add_scriptconfig_arguments(
                config_cls=config_cls,
                added_args=added_args,
                function_or_class=function_or_class,
                component=component,
                container=container,
                nested_key=nested_key,
                fail_untyped=fail_untyped,
                as_positional=as_positional,
                sub_configs=sub_configs,
                skip=skip,
                linked_targets=linked_targets,
            )

        return added_args

    def _add_scriptconfig_arguments(
        self,
        config_cls,
        added_args,
        function_or_class,
        component,
        container,
        nested_key,
        fail_untyped: bool = True,
        as_positional: bool = False,
        sub_configs: bool = False,
        skip: Optional[Set[Union[str, int]]] = None,
        linked_targets: Optional[Set[str]] = None,
    ):

        for key, value in config_cls.__default__.items():
            from scriptconfig import value as value_mod

            if not isinstance(value, value_mod.Value):
                # hack
                value = value_mod.Value(value)
            type_ = value.parsekw["type"]
            if type_ is None or not isinstance(type_, type):
                annotation = inspect._empty
            else:
                annotation = type_
            param = ParamData_Extension(
                name=key,
                annotation=annotation,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=value.value,
                doc=value.parsekw["help"],
                component=component,
                parent=function_or_class,
                short_aliases=value.short_alias,
                long_aliases=value.alias,
            )
            self._add_signature_parameter(
                container,
                nested_key,
                param,
                added_args,
                skip={s for s in (skip or []) if isinstance(s, str)},
                fail_untyped=fail_untyped,
                sub_configs=sub_configs,
                linked_targets=linked_targets,
                as_positional=as_positional,
            )

    def _add_signature_parameter(
        self,
        container,
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
        **kwargs,
    ):
        name = param.name
        kind = param.kind
        annotation = param.annotation
        if default == inspect_empty:
            default = param.default
            if default == inspect_empty and is_optional(annotation):
                default = None
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
        if (
            kind in {kinds.VAR_POSITIONAL, kinds.VAR_KEYWORD}
            or (not is_required and name[0] == "_")
            or (annotation == inspect_empty and not is_required and default is None)
        ):
            return
        elif skip and name in skip:
            self.logger.debug(skip_message + "Parameter requested to be skipped.")
            return
        if is_factory_class(default):
            default = param.parent.__dataclass_fields__[name].default_factory()
        if annotation == inspect_empty and not is_required:
            annotation = Union[type(default), Any]
        if "help" not in kwargs:
            kwargs["help"] = param.doc
        if not is_required:
            kwargs["default"] = default
            if default is None and not is_optional(annotation, object):
                annotation = Optional[annotation]
        elif not as_positional:
            kwargs["required"] = True
        is_subclass_typehint = False
        is_dataclass_like_typehint = is_dataclass_like(annotation)
        # dest = (nested_key + "." if nested_key else "") + name
        # args = [dest if is_required and as_positional else "--" + dest]
        args, dest = param._resolve_args_and_dest(is_required, as_positional, nested_key)
        if param.origin:
            parser = container
            if not isinstance(container, ArgumentParser):
                parser = getattr(container, "parser")
            group_name = "; ".join(str(o) for o in param.origin)
            if group_name in parser.groups:
                container = parser.groups[group_name]
            else:
                container = parser.add_argument_group(
                    f"Conditional arguments [origins: {group_name}]",
                    name=group_name,
                )
        if (
            annotation in {str, int, float, bool}
            or is_subclass(annotation, (str, int, float))
            or is_dataclass_like_typehint
        ):
            kwargs["type"] = annotation
            register_pydantic_type(annotation)
        elif annotation != inspect_empty:
            try:
                is_subclass_typehint = ActionTypeHint.is_subclass_typehint(annotation, all_subtypes=False)
                kwargs["type"] = annotation
                sub_add_kwargs: dict = {"fail_untyped": fail_untyped, "sub_configs": sub_configs}
                if is_subclass_typehint:
                    prefix = f"{name}.init_args."
                    subclass_skip = {s[len(prefix) :] for s in skip or [] if s.startswith(prefix)}
                    sub_add_kwargs["skip"] = subclass_skip
                else:
                    register_pydantic_type(annotation)
                enable_path = sub_configs and (
                    is_subclass_typehint or ActionTypeHint.is_return_subclass_typehint(annotation)
                )
                args = ActionTypeHint.prepare_add_argument(
                    args=args,
                    kwargs=kwargs,
                    enable_path=enable_path,
                    container=container,
                    logger=self.logger,
                    sub_add_kwargs=sub_add_kwargs,
                )
            except ValueError as ex:
                self.logger.debug(skip_message + str(ex))
        if "type" in kwargs or "action" in kwargs:
            sub_add_kwargs = {
                "fail_untyped": fail_untyped,
                "sub_configs": sub_configs,
                "instantiate": instantiate,
            }
            if is_dataclass_like_typehint:
                kwargs.update(sub_add_kwargs)
            with ActionTypeHint.allow_default_instance_context():
                action = container.add_argument(*args, **kwargs)
            action.sub_add_kwargs = sub_add_kwargs
            if is_subclass_typehint and len(subclass_skip) > 0:
                action.sub_add_kwargs["skip"] = subclass_skip
            added_args.append(dest)
        elif is_required and fail_untyped:
            raise ValueError(
                "With fail_untyped=True, all mandatory parameters must have a supported"
                f" type. Parameter '{name}' from '{src}' does not specify a type."
            )


def is_factory_class(value):
    return value.__class__ == dataclasses._HAS_DEFAULT_FACTORY_CLASS


def raise_unexpected_value(message: str, val: Any = inspect._empty, exception: Optional[Exception] = None):
    """
    Patch to disable overzelous type hints
    """
    if val is not inspect._empty:
        message += f". Got value: {val} (note: jsonargparse warnings can be overzealous)"
    import warnings
    warnings.warn(message)
    # raise ValueError(message) from exception


def apply_monkeypatch():
    """
    Dynamically add the monkeypatch to jsonargparse
    """
    import jsonargparse
    # for key in jsonargparse.__all__:
    #     value = getattr(jsonargparse, key)
    #     if isinstance(value, type) and issubclass(value, SignatureArguments):

    # Hack in the new methods from SignatureArguments
    jsonargparse._signatures.ParamData = ParamData_Extension
    jsonargparse._parameter_resolvers.ParamData = ParamData_Extension
    jsonargparse._signatures.SignatureArguments._add_signature_arguments = SignatureArguments_Extension._add_signature_arguments
    jsonargparse._signatures.SignatureArguments._add_signature_parameter = SignatureArguments_Extension._add_signature_parameter
    jsonargparse._signatures.SignatureArguments._add_scriptconfig_arguments = SignatureArguments_Extension._add_scriptconfig_arguments

    # hack to disable config checks
    import ubelt as ub
    jsonargparse._core.ArgumentParser.check_config = ub.identity
    jsonargparse._typehints.raise_unexpected_value = raise_unexpected_value
