"""
Utilities to derive what the scriptconfig arguments should look like based on
function signatures. This code is taken and modified from jsonargparse.

It is probably far more than we actually need here, and it would be nice if we
could just use it as a library, but I don't want to depend on non-public APIs.

References:
    https://github.com/omni-us/jsonargparse/blob/2ba4e5fedaebd17a9a0dddf648076a719b78cf09/jsonargparse/_parameter_resolvers.py

Ignore:
    from jsonargparse import _parameter_resolvers
    import liberator
    from jsonargparse._typehints import ActionTypeHint, get_optional_arg, get_subclass_types
    lib = liberator.Liberator()
    lib.add_dynamic(_parameter_resolvers.get_signature_parameters)
    lib.add_dynamic(get_subclass_types)
    lib.expand(['jsonargparse'])
    lib.current_sourcecode()

Example:
    >>> # xdoctest: +SKIP("not ready")
    >>> from geowatch.tasks.fusion import _jsonargparse_introspect
    >>> import ubelt as ub
    >>> function_or_class = _jsonargparse_introspect.get_signature_parameters
    >>> params = _jsonargparse_introspect.get_signature_parameters(function_or_class)
    >>> print(f'params = {ub.urepr(params, nl=1)}')

Ignore:
    >>> # xdoctest: +SKIP("not ready")
    >>> from scriptconfig.introspection.complex_introspection import *  # NOQA
    >>> from lightning import Trainer
    >>> import ubelt as ub
    >>> params = get_signature_parameters(Trainer)
    >>> print(f'params = {ub.urepr(params, nl=1)}')

    import scriptconfig as scfg
    default = {}
    for param in params:
        default[param.name] = scfg.Value(param.default, help=param.doc)

    class TrainerConfig(scfg.DataConfig):
        __default__ = default

    text = TrainerConfig.port_to_dataconf(TrainerConfig())
    print(text)

    config = TrainerConfig()
    import kwutil
    yaml_text = kwutil.Yaml.dumps(dict(config), backend='pyyaml')
    print(yaml_text)

    lines = []
    for key, value in config.items():
        doc = ub.paragraph(TrainerConfig.__default__[key].help)
        line = (f'{key}: {value} # {doc}')
        lines.append(line)
    text = chr(10).join(lines)
    text = ub.util_repr._align_text(text, ':')
    text = ub.util_repr._align_text(text, '#')
    print(text)
"""

from collections import OrderedDict
from collections import abc
from collections import defaultdict
from collections import namedtuple
from contextlib import contextmanager
from contextlib import suppress
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import is_dataclass
from functools import partial
from importlib import import_module
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from types import MappingProxyType
from types import MethodType
from typing import Any
from typing import Callable
from typing import Dict
from typing import ForwardRef
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import MutableSet
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union
from typing import _GenericAlias
from typing import get_type_hints
import ast
import dataclasses
import inspect
import logging
import os
import sys
import textwrap


@dataclasses.dataclass
class ParamData:
    name: str
    annotation: Any
    default: Any = inspect._empty
    kind: Optional[inspect._ParameterKind] = None
    doc: Optional[str] = None
    component: Optional[Union[Callable, Type, Tuple]] = None
    parent: Optional[Union[Type, Tuple]] = None
    origin: Optional[Union[str, Tuple]] = None


ParamList = List[ParamData]


def is_generic_class(cls) -> bool:
    return isinstance(cls, _GenericAlias) and getattr(cls, "__module__", "") != "typing"


dump_json_kwargs = {
    "ensure_ascii": False,
    "sort_keys": False,
}


def json_compact_dump(data):
    import json

    return json.dumps(data, separators=(",", ":"), **dump_json_kwargs)


def hash_item(item):
    try:
        if isinstance(item, (dict, list)):
            item_hash = hash(json_compact_dump(item))
        else:
            item_hash = hash(item)
    except Exception:
        item_hash = hash(repr(item))
    return item_hash


def unique(iterable):
    unique_items = []
    seen = set()
    for item in iterable:
        key = hash_item(item)
        if key not in seen:
            unique_items.append(item)
            seen.add(key)
    return unique_items


def iter_to_set_str(val, sep=","):
    val = unique(val)
    if len(val) == 1:
        return str(val[0])
    return "{" + sep.join(str(x) for x in val) + "}"


unpack_meta_types = set()


def is_unpack_typehint(cls) -> bool:
    return any(isinstance(cls, unpack_type) for unpack_type in unpack_meta_types)


def is_final_class(cls) -> bool:
    """Checks whether a class is final, i.e. decorated with ``typing.final``."""
    return getattr(cls, "__final__", False)


def is_dataclass_like(cls) -> bool:
    if is_generic_class(cls):
        return is_dataclass_like(cls.__origin__)
    if not inspect.isclass(cls) or cls is object:
        return False
    if is_final_class(cls):
        return True
    classes = [c for c in inspect.getmro(cls) if c not in {object, Generic}]
    all_dataclasses = all(dataclasses.is_dataclass(c) for c in classes)

    if not all_dataclasses:
        raise NotImplementedError('liberator did not resolve internal imports')
        from ._optionals import attrs_support, is_pydantic_model

        if is_pydantic_model(cls):
            return True

        if attrs_support:
            import attrs

            if attrs.has(cls):
                return True

    return all_dataclasses


typing_extensions_support = find_spec("typing_extensions") is not None


def typing_extensions_import(name):
    if typing_extensions_support:
        return getattr(__import__("typing_extensions"), name, False)
    else:
        return getattr(__import__("typing"), name, False)


annotated_alias = typing_extensions_import("_AnnotatedAlias")


def is_annotated(typehint: type) -> bool:
    return annotated_alias and isinstance(typehint, annotated_alias)


def get_generic_origin(cls):
    return cls.__origin__ if is_generic_class(cls) else cls


unresolvable_import_paths = {}


def get_module_var_path(module_path: str, value: Any) -> Optional[str]:
    module = import_module(module_path)
    for name, var in vars(module).items():
        if var is value:
            return module_path + "." + name
    return None


def get_import_path(value: Any) -> Optional[str]:
    """Returns the shortest dot import path for the given object."""
    path = None
    value = get_generic_origin(value)
    if (
        hasattr(value, "__self__")
        and inspect.isclass(value.__self__)
        and inspect.ismethod(value)
    ):
        module_path = getattr(value.__self__, "__module__", None)
        qualname = f"{value.__self__.__name__}.{value.__name__}"
    else:
        module_path = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", "")

    if module_path is None:
        path = unresolvable_import_paths.get(value)
        if path:
            module_path, _ = path.rsplit(".", 1)
    elif (not qualname and not inspect.isclass(value)) or (
        inspect.ismethod(value) and not inspect.isclass(value.__self__)
    ):
        path = get_module_var_path(module_path, value)
    elif qualname:
        path = module_path + "." + qualname

    if not path:
        raise ValueError(
            f"Not possible to determine the import path for object {value}."
        )

    if qualname and module_path and ("." in qualname or "." in module_path):
        module_parts = module_path.split(".")
        for num in range(len(module_parts)):
            module_path = ".".join(module_parts[: num + 1])
            module = import_module(module_path)
            if "." in qualname:
                obj_name, attr = qualname.rsplit(".", 1)
                obj = getattr(module, obj_name, None)
                if getattr(module, attr, None) is value:
                    path = module_path + "." + attr
                    break
                elif getattr(obj, attr, None) == value:
                    path = module_path + "." + qualname
                    break
            elif getattr(module, qualname, None) is value:
                path = module_path + "." + qualname
                break
    return path


def get_typehint_origin(typehint):
    if not hasattr(typehint, "__origin__"):
        typehint_class = get_import_path(typehint.__class__)
        if typehint_class == "types.UnionType":
            return Union
        if typehint_class in {
            "typing._TypedDictMeta",
            "typing_extensions._TypedDictMeta",
        }:
            return dict
    return getattr(typehint, "__origin__", None)


typeshed_client_support = find_spec("typeshed_client") is not None


def import_typeshed_client():
    if typeshed_client_support:
        import typeshed_client

        return typeshed_client
    else:
        return __import__("argparse").Namespace(
            ImportedInfo=object, ModulePath=object, Resolver=object
        )


class NamesVisitor(ast.NodeVisitor):
    def visit_Name(self, node: ast.Name) -> None:
        self.names_found.append(node.id)

    def find(self, node: ast.AST) -> list:
        self.names_found: List[str] = []
        self.visit(node)
        self.names_found = unique(self.names_found)
        return self.names_found


var_map = namedtuple("var_map", "name value")


union_map = var_map(name="Union", value=Union)


pep585_map = {
    "dict": var_map(name="Dict", value=Dict),
    "frozenset": var_map(name="FrozenSet", value=FrozenSet),
    "list": var_map(name="List", value=List),
    "set": var_map(name="Set", value=Set),
    "tuple": var_map(name="Tuple", value=Tuple),
    "type": var_map(name="Type", value=Type),
}


none_map = var_map(name="NoneType", value=type(None))


class BackportTypeHints(ast.NodeTransformer):
    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        if isinstance(node.value, ast.Name) and node.value.id in pep585_map:
            value = self.new_name_load(pep585_map[node.value.id])
        else:
            value = node.value  # type: ignore[assignment]
        return ast.Subscript(
            value=value,
            slice=self.visit(node.slice),
            ctx=ast.Load(),
        )

    def visit_Constant(self, node: ast.Constant) -> Union[ast.Constant, ast.Name]:
        if node.value is None:
            return self.new_name_load(none_map)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> Union[ast.BinOp, ast.Subscript]:
        out_node: Union[ast.BinOp, ast.Subscript] = node
        if isinstance(node.op, ast.BitOr):
            elts: list = []
            self.append_union_elts(node.left, elts)
            self.append_union_elts(node.right, elts)
            out_node = ast.Subscript(
                value=self.new_name_load(union_map),
                slice=ast.Index(  # type: ignore[arg-type,call-arg]
                    value=ast.Tuple(elts=elts, ctx=ast.Load()),
                    ctx=ast.Load(),
                ),
                ctx=ast.Load(),
            )
        return out_node

    def append_union_elts(self, node: ast.AST, elts: list) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            self.append_union_elts(node.left, elts)
            self.append_union_elts(node.right, elts)
        else:
            elts.append(self.visit(node))

    def new_name_load(self, var: var_map) -> ast.Name:
        name = f"_{self.__class__.__name__}_{var.name}"
        self.exec_vars[name] = var.value
        return ast.Name(id=name, ctx=ast.Load())

    def backport(self, input_ast: ast.AST, exec_vars: dict) -> ast.AST:
        typing = __import__("typing")
        for key, value in exec_vars.items():
            if getattr(value, "__module__", "") == "collections.abc":
                if hasattr(typing, key):
                    exec_vars[key] = getattr(typing, key)
        self.exec_vars = exec_vars
        backport_ast = self.visit(deepcopy(input_ast))
        return ast.fix_missing_locations(backport_ast)


def get_arg_type(arg_ast, aliases):
    type_ast = ast.parse("___arg_type___ = 0")
    type_ast.body[0].value = arg_ast
    exec_vars = {}
    bad_aliases = {}
    add_asts = False
    for name in NamesVisitor().find(arg_ast):
        value = aliases[name]
        if isinstance(value, tuple):
            value = value[1]
        if isinstance(value, Exception):
            bad_aliases[name] = value
        elif isinstance(value, ast.AST):
            add_asts = True
        else:
            exec_vars[name] = value
    if add_asts:
        body = []
        for name, (_, value) in aliases.items():
            if isinstance(value, ast.AST):
                body.append(ast.fix_missing_locations(value))
            elif not isinstance(value, Exception):
                exec_vars[name] = value
        type_ast.body = body + type_ast.body
        if "TypeAlias" not in exec_vars:
            type_alias = typing_extensions_import("TypeAlias")
            if type_alias:
                exec_vars["TypeAlias"] = type_alias
    if sys.version_info < (3, 10):
        backporter = BackportTypeHints()
        type_ast = backporter.backport(type_ast, exec_vars)
    try:
        exec(compile(type_ast, filename="<ast>", mode="exec"), exec_vars, exec_vars)
    except NameError as ex:
        ex_from = None
        for name, alias_exception in bad_aliases.items():
            if str(ex) == f"name '{name}' is not defined":
                ex_from = alias_exception
                break
        raise ex from ex_from
    return exec_vars["___arg_type___"]


tc = import_typeshed_client()


def ast_annassign_to_assign(node: ast.AnnAssign) -> ast.Assign:
    return ast.Assign(
        targets=[node.target],
        value=node.value,  # type: ignore[arg-type]
        type_ignores=[],  # type: ignore[call-arg]
        lineno=node.lineno,
        end_lineno=node.lineno,
    )


def import_module_or_none(path: str):
    if path.endswith(".__init__"):
        path = path[:-9]
    try:
        return import_module(path)
    except ModuleNotFoundError:
        return None


def get_source_module(path: str, component) -> tc.ModulePath:
    if component is None:
        module_path, name = path.rsplit(".", 1)
        component = getattr(import_module_or_none(module_path), name, None)
    if component is not None:
        module = inspect.getmodule(component)
        assert module is not None
        module_path = module.__name__
        if getattr(module, "__file__", "").endswith("__init__.py"):
            module_path += ".__init__"
    return tc.ModulePath(tuple(module_path.split(".")))


def get_mro_method_parent(parent, method_name):
    while hasattr(parent, "__dict__") and method_name not in parent.__dict__:
        try:
            parent = inspect.getmro(parent)[1]
        except IndexError:
            parent = None
    return None if parent is object else parent


def alias_is_unique(aliases, name, source, value):
    if name in aliases:
        src, val = aliases[name]
        if src != source:
            return val is value
    return True


def alias_already_added(aliases, name, source):
    return name in aliases and aliases[name][0] in {"__builtins__", source}


class MethodsVisitor(ast.NodeVisitor):
    method_found: Optional[ast.FunctionDef]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self.method_found and node.name == self.method_name:
            self.method_found = node

    def visit_If(self, node: ast.If) -> None:
        test_ast = ast.parse("___test___ = 0")
        test_ast.body[0].value = node.test  # type: ignore[attr-defined]
        exec_vars = {"sys": sys}
        with suppress(Exception):
            exec(compile(test_ast, filename="<ast>", mode="exec"), exec_vars, exec_vars)
            if exec_vars["___test___"]:
                node.orelse = []
            else:
                node.body = []
            self.generic_visit(node)

    def find(self, node: ast.AST, method_name: str) -> Optional[ast.FunctionDef]:
        self.method_name = method_name
        self.method_found = None
        self.visit(node)
        return self.method_found


class ImportsVisitor(ast.NodeVisitor):
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            module_path = self.module_path[: -node.level]
            if node.module:
                module_path.append(node.module)
            node = deepcopy(node)
            node.module = ".".join(module_path)
            node.level = 0
        for alias in node.names:
            self.imports_found[alias.asname or alias.name] = (node.module, alias.name)

    def find(
        self, node: ast.AST, module_path: str
    ) -> Dict[str, Tuple[Optional[str], str]]:
        self.module_path = module_path.split(".")
        self.imports_found: Dict[str, Tuple[Optional[str], str]] = {}
        self.visit(node)
        return self.imports_found


class AssignsVisitor(ast.NodeVisitor):
    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if hasattr(target, "id"):
                self.assigns_found[target.id] = node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if hasattr(node.target, "id"):
            self.assigns_found[node.target.id] = ast_annassign_to_assign(node)

    def find(self, node: ast.AST) -> Dict[str, ast.Assign]:
        self.assigns_found: Dict[str, ast.Assign] = {}
        self.visit(node)
        return self.assigns_found


class StubsResolver(tc.Resolver):
    def __init__(self, search_context=None) -> None:
        super().__init__(search_context)
        self._module_ast_cache: Dict[str, Optional[ast.AST]] = {}
        self._module_assigns_cache: Dict[str, Dict[str, ast.Assign]] = {}
        self._module_imports_cache: Dict[str, Dict[str, Tuple[Optional[str], str]]] = {}

    def get_imported_info(self, path: str, component=None) -> Optional[tc.ImportedInfo]:
        resolved = self.get_fully_qualified_name(path)
        imported_info = None
        if isinstance(resolved, tc.ImportedInfo):
            resolved = resolved.info
        if isinstance(resolved, tc.NameInfo):
            source_module = get_source_module(path, component)
            imported_info = tc.ImportedInfo(source_module=source_module, info=resolved)
        return imported_info

    def get_component_imported_info(
        self, component, parent
    ) -> Optional[tc.ImportedInfo]:
        if not parent and inspect.ismethod(component):
            parent = type(component.__self__)
            component = getattr(parent, component.__name__)
        if not parent:
            return self.get_imported_info(
                f"{component.__module__}.{component.__name__}", component
            )
        parent = get_mro_method_parent(parent, component.__name__)
        stub_import = parent and self.get_imported_info(
            f"{parent.__module__}.{parent.__name__}", component
        )
        if stub_import and isinstance(stub_import.info.ast, ast.AST):
            method_ast = MethodsVisitor().find(stub_import.info.ast, component.__name__)
            if method_ast is None:
                stub_import = None
            else:
                name_info = tc.NameInfo(
                    name=component.__qualname__, is_exported=False, ast=method_ast
                )
                stub_import = tc.ImportedInfo(
                    source_module=stub_import.source_module, info=name_info
                )
        return stub_import

    def get_aliases(self, imported_info: tc.ImportedInfo):
        aliases: Dict[str, Tuple[str, Any]] = {}
        self.add_import_aliases(aliases, imported_info)
        return aliases

    def get_module_stub_ast(self, module_path: str):
        if module_path not in self._module_ast_cache:
            self._module_ast_cache[module_path] = tc.get_stub_ast(
                module_path, search_context=self.ctx
            )
        return self._module_ast_cache[module_path]

    def get_module_stub_assigns(self, module_path: str):
        if module_path not in self._module_assigns_cache:
            module_ast = self.get_module_stub_ast(module_path)
            self._module_assigns_cache[module_path] = AssignsVisitor().find(module_ast)
        return self._module_assigns_cache[module_path]

    def get_module_stub_imports(self, module_path: str):
        if module_path not in self._module_imports_cache:
            module_ast = self.get_module_stub_ast(module_path)
            self._module_imports_cache[module_path] = ImportsVisitor().find(
                module_ast, module_path
            )
        return self._module_imports_cache[module_path]

    def add_import_aliases(self, aliases, stub_import: tc.ImportedInfo):
        module_path = ".".join(stub_import.source_module)
        module = import_module_or_none(module_path)
        stub_ast: Optional[ast.AST] = None
        if isinstance(stub_import.info.ast, (ast.Assign, ast.AnnAssign)):
            stub_ast = stub_import.info.ast.value
        elif isinstance(stub_import.info.ast, ast.AST):
            stub_ast = stub_import.info.ast
        if stub_ast:
            self.add_module_aliases(aliases, module_path, module, stub_ast)
        return module_path, stub_import.info.ast

    def add_module_aliases(self, aliases, module_path, module, node):
        names = NamesVisitor().find(node) if node else []
        for name in names:
            if alias_already_added(aliases, name, module_path):
                continue
            source = module_path
            if name in __builtins__:
                source = "__builtins__"
                value = __builtins__[name]
            elif hasattr(module, name):
                value = getattr(module, name)
            elif name in self.get_module_stub_assigns(module_path):
                value = self.get_module_stub_assigns(module_path)[name]
                self.add_module_aliases(aliases, module_path, module, value.value)
            elif name in self.get_module_stub_imports(module_path):
                imported_module_path, imported_name = self.get_module_stub_imports(
                    module_path
                )[name]
                imported_module = import_module_or_none(imported_module_path)
                if hasattr(imported_module, imported_name):
                    source = imported_module_path
                    value = getattr(imported_module, imported_name)
                else:
                    stub_import = self.get_imported_info(
                        f"{imported_module_path}.{imported_name}"
                    )
                    source, value = self.add_import_aliases(aliases, stub_import)
            else:
                value = NotImplementedError(
                    f"{name!r} from {module_path!r} not in builtins, module or stub"
                )
            if alias_already_added(aliases, name, source):
                continue
            if not alias_is_unique(aliases, name, source, value):
                value = NotImplementedError(
                    f"non-unique alias {name!r}: {aliases[name][1]} ({aliases[name][0]}) vs {value} ({source})"
                )
            aliases[name] = (source, value)


kinds = inspect._ParameterKind

stubs_resolver = None


def get_stubs_resolver():
    global stubs_resolver
    if not stubs_resolver:
        search_path = [Path(p) for p in sys.path]
        search_context = tc.get_search_context(search_path=search_path)
        stubs_resolver = StubsResolver(search_context=search_context)
    return stubs_resolver


def get_stub_types(params, component, parent, logger) -> Optional[Dict[str, Any]]:
    if not typeshed_client_support:
        return None
    missing_types = {
        p.name: n
        for n, p in enumerate(params)
        if p.kind not in {kinds.VAR_POSITIONAL, kinds.VAR_KEYWORD}
        and p.annotation == inspect._empty
    }
    if not missing_types:
        return None
    resolver = get_stubs_resolver()
    stub_import = resolver.get_component_imported_info(component, parent)
    if not stub_import:
        return None
    known_params = {p.name for p in params}
    aliases = resolver.get_aliases(stub_import)
    arg_asts = stub_import.info.ast.args.args + stub_import.info.ast.args.kwonlyargs
    types = {}
    for arg_ast in arg_asts[1:] if parent else arg_asts:
        name = arg_ast.arg
        if arg_ast.annotation and (name in missing_types or name not in known_params):
            try:
                types[name] = get_arg_type(arg_ast.annotation, aliases)
            except Exception as ex:
                logger.debug(
                    f"Failed to parse type stub for {component.__qualname__!r} parameter {name!r}",
                    exc_info=ex,
                )
                if name not in known_params:
                    types[name] = inspect._empty  # pragma: no cover
    return types


def get_annotated_base_type(typehint: type) -> type:
    return typehint.__origin__  # type: ignore[attr-defined]


class ClassFromFunctionBase:
    wrapped_function: Callable


@contextmanager
def missing_package_raise(package, importer):
    try:
        yield None
    except ImportError as ex:
        raise ImportError(
            f"{package} package is required by {importer} :: {ex}"
        ) from ex


_docstring_parse_options = {
    "style": None,
    "attribute_docstrings": False,
}


def import_docstring_parser(importer):
    with missing_package_raise("docstring-parser", importer):
        import docstring_parser
    return docstring_parser


def get_docstring_parse_options():
    if _docstring_parse_options["style"] is None:
        dp = import_docstring_parser("get_docstring_parse_options")
        _docstring_parse_options["style"] = dp.DocstringStyle.AUTO
    return _docstring_parse_options


def parse_docstring(component, params=False, logger=None):
    dp = import_docstring_parser("parse_docstring")
    options = get_docstring_parse_options()
    try:
        if params and options["attribute_docstrings"]:
            return dp.parse_from_object(component, style=options["style"])
        else:
            return dp.parse(component.__doc__, style=options["style"])
    except (ValueError, dp.ParseError) as ex:
        if logger:
            logger.debug(f"Failed parsing docstring for {component}: {ex}")
    return None


docstring_parser_support = find_spec("docstring_parser") is not None


def parse_docs(component, parent, logger):
    docs = {}
    if docstring_parser_support:
        doc_sources = [component]
        if inspect.isclass(parent) and component.__name__ == "__init__":
            doc_sources += [parent]
        for src in doc_sources:
            doc = parse_docstring(src, params=True, logger=logger)
            if doc:
                for param in doc.params:
                    docs[param.arg_name] = param.description
    return docs


def is_subclass(cls, class_or_tuple) -> bool:
    """Extension of issubclass that supports non-class arguments."""
    try:
        return inspect.isclass(cls) and issubclass(cls, class_or_tuple)
    except TypeError:
        return False


def get_pydantic_support() -> int:
    support = "0"
    if find_spec("pydantic"):
        support = version("pydantic")
    return int(support.split(".", 1)[0])


pydantic_support = get_pydantic_support()


def is_pydantic_model(class_type) -> int:
    classes = (
        inspect.getmro(class_type)
        if pydantic_support and inspect.isclass(class_type)
        else []
    )
    for cls in classes:
        if (
            getattr(cls, "__module__", "").startswith("pydantic")
            and getattr(cls, "__name__", "") == "BaseModel"
        ):
            import pydantic

            if issubclass(cls, pydantic.BaseModel):
                return pydantic_support
            elif pydantic_support > 1 and issubclass(cls, pydantic.v1.BaseModel):
                return 1
    return 0


type_alias_type = typing_extensions_import("TypeAliasType")


def is_alias_type(typehint: type) -> bool:
    return type_alias_type and isinstance(typehint, type_alias_type)


def get_alias_target(typehint: type) -> bool:
    return typehint.__value__  # type: ignore[attr-defined]


def get_unaliased_type(cls):
    new_cls = cls
    while True:
        cur_cls = new_cls
        if is_annotated(new_cls):
            new_cls = get_annotated_base_type(new_cls)
        if is_alias_type(new_cls):
            new_cls = get_alias_target(new_cls)
        if new_cls == cur_cls:
            break
    return cur_cls


tuple_set_origin_types = {
    Tuple,
    tuple,
    Set,
    set,
    frozenset,
    MutableSet,
    abc.Set,
    abc.MutableSet,
}


sequence_origin_types = {
    List,
    list,
    Iterable,
    Sequence,
    MutableSequence,
    abc.Iterable,
    abc.Sequence,
    abc.MutableSequence,
}


mapping_origin_types = {
    Dict,
    dict,
    Mapping,
    MappingProxyType,
    MutableMapping,
    abc.Mapping,
    abc.MutableMapping,
    OrderedDict,
}


def getattr_recursive(obj, attr):
    if "." in attr:
        attr, *attrs = attr.split(".", 1)
        return getattr_recursive(getattr(obj, attr), attrs[0])
    return getattr(obj, attr)


class TypeCheckingVisitor(ast.NodeVisitor):
    type_checking_names: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "typing":
                name = ast.dump(
                    ast.Attribute(
                        value=ast.Name(id=alias.asname or "typing", ctx=ast.Load()),
                        attr="TYPE_CHECKING",
                        ctx=ast.Load(),
                    )
                )
                self.type_checking_names.append(name)
                break

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "typing":
            for alias in node.names:
                if alias.name == "TYPE_CHECKING":
                    name = ast.dump(
                        ast.Name(id=alias.asname or "TYPE_CHECKING", ctx=ast.Load())
                    )
                    self.type_checking_names.append(name)
                    break

    def visit_If(self, node: ast.If) -> None:
        if (
            isinstance(node.test, (ast.Name, ast.Attribute))
            and any(ast.dump(node.test) == n for n in self.type_checking_names)
        ) or (
            isinstance(node.test, ast.BoolOp)
            and isinstance(node.test.op, (ast.And, ast.Or))
            and any(
                ast.dump(v) == n
                for n in self.type_checking_names
                for v in node.test.values
            )
        ):
            ast_exec = ast.parse("")
            ast_exec.body = node.body
            try:
                exec(
                    compile(ast_exec, filename="<ast>", mode="exec"),
                    self.aliases,
                    self.aliases,
                )
            except Exception as ex:
                if self.logger:
                    self.logger.debug(
                        f"Failed to execute 'TYPE_CHECKING' block in '{self.module}'",
                        exc_info=ex,
                    )

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.If, ast.Module)):
            super().generic_visit(node)

    def update_aliases(
        self,
        module_source: str,
        module: str,
        aliases: dict,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.module = module
        self.aliases = aliases
        self.logger = logger
        module_tree = ast.parse(module_source)
        self.visit(module_tree)


def resolve_forward_refs(arg_type, aliases, logger):
    if isinstance(arg_type, str) and arg_type in aliases:
        arg_type = aliases[arg_type]

    def resolve_subtypes_forward_refs(typehint):
        if has_subtypes(typehint):
            try:
                subtypes = []
                for arg in typehint.__args__:
                    if isinstance(arg, ForwardRef):
                        forward_arg, *forward_args = arg.__forward_arg__.split(".", 1)
                        if forward_arg in aliases:
                            arg = aliases[forward_arg]
                            if forward_args:
                                arg = getattr_recursive(arg, forward_args[0])
                        else:
                            raise NameError(f"Name '{forward_arg}' is not defined")
                    else:
                        arg = resolve_subtypes_forward_refs(arg)
                    subtypes.append(arg)
                if subtypes != list(typehint.__args__):
                    typehint_origin = get_typehint_origin(typehint)
                    if sys.version_info < (3, 10):
                        if typehint_origin in sequence_origin_types:
                            typehint_origin = List
                        elif typehint_origin in tuple_set_origin_types:
                            typehint_origin = Tuple
                        elif typehint_origin in mapping_origin_types:
                            typehint_origin = Dict
                        elif typehint_origin == type:
                            typehint_origin = Type
                    typehint = typehint_origin[tuple(subtypes)]
            except Exception as ex:
                if logger:
                    logger.debug(
                        f"Failed to resolve forward refs in {typehint}", exc_info=ex
                    )
        return typehint

    return resolve_subtypes_forward_refs(arg_type)


def has_subtypes(typehint):
    typehint_origin = get_typehint_origin(typehint)
    if typehint_origin is type and hasattr(typehint, "__args__"):
        return True

    return (
        typehint_origin == Union
        or typehint_origin in sequence_origin_types
        or typehint_origin in tuple_set_origin_types
        or typehint_origin in mapping_origin_types
    )


def get_global_vars(obj: Any, logger: Optional[logging.Logger]) -> dict:
    global_vars = vars(import_module(obj.__module__))
    try:
        module_source = (
            inspect.getsource(sys.modules[obj.__module__])
            if obj.__module__ in sys.modules
            else ""
        )
        if "TYPE_CHECKING" in module_source:
            TypeCheckingVisitor().update_aliases(
                module_source, obj.__module__, global_vars, logger
            )
    except Exception as ex:
        if logger:
            logger.debug(
                f"Failed to update aliases for TYPE_CHECKING blocks in {obj.__module__}",
                exc_info=ex,
            )
    return global_vars


def type_requires_eval(typehint):
    if has_subtypes(typehint):
        return any(type_requires_eval(a) for a in getattr(typehint, "__args__", []))
    return isinstance(typehint, (str, ForwardRef))


def get_types(obj: Any, logger: Optional[logging.Logger] = None) -> dict:
    global_vars = get_global_vars(obj, logger)
    try:
        types = get_type_hints(obj, global_vars)
    except Exception as ex1:
        types = ex1  # type: ignore[assignment]
    if isinstance(types, dict) and all(
        not type_requires_eval(t) for t in types.values()
    ):
        return types

    try:
        source = textwrap.dedent(inspect.getsource(obj))
        tree = ast.parse(source)
        assert isinstance(tree, ast.Module) and len(tree.body) == 1
        node = tree.body[0]
        assert isinstance(node, (ast.FunctionDef, ast.ClassDef))
    except Exception as ex2:
        if isinstance(types, Exception):
            if logger:
                logger.debug(f"Failed to parse to source code for {obj}", exc_info=ex2)
            raise type(types)(f"{repr(types)} + {repr(ex2)}") from ex2  # type: ignore[arg-type]
        return types

    aliases = __builtins__.copy()  # type: ignore[attr-defined]
    aliases.update(global_vars)
    ex = None
    if isinstance(types, Exception):
        ex = types
        types = {}

    if isinstance(node, ast.FunctionDef):
        arg_asts = [
            (a.arg, a.annotation) for a in node.args.args + node.args.kwonlyargs
        ]
    else:
        arg_asts = [(a.target.id, a.annotation) for a in node.body if isinstance(a, ast.AnnAssign)]  # type: ignore[union-attr]

    for name, annotation in arg_asts:
        if annotation and (name not in types or type_requires_eval(types[name])):
            try:
                if isinstance(annotation, ast.Constant) and annotation.value in aliases:
                    types[name] = aliases[annotation.value]
                else:
                    arg_type = get_arg_type(annotation, aliases)
                    types[name] = resolve_forward_refs(arg_type, aliases, logger)
            except Exception as ex3:
                types[name] = ex3

    if all(isinstance(t, Exception) for t in types.values()):
        raise ex or next(iter(types.values()))

    return types


def evaluate_postponed_annotations(params, component, parent, logger):
    if not (params and any(type_requires_eval(p.annotation) for p in params)):
        return
    try:
        if (
            is_dataclass(parent)
            and component.__name__ == "__init__"
            and not component.__qualname__.startswith(parent.__name__ + ".")
        ):
            types = get_types(parent, logger)
        else:
            types = get_types(component, logger)
    except Exception as ex:
        logger.debug(f"Unable to evaluate types for {component}", exc_info=ex)
        return
    for param in params:
        if param.name in types:
            param_type = types[param.name]
            if isinstance(param_type, Exception):
                logger.debug(
                    f"Unable to evaluate type of {param.name} from {component}",
                    exc_info=param_type,
                )
                continue
            param.annotation = param_type


reconplogger_support = find_spec("reconplogger") is not None


def import_reconplogger(importer):
    with missing_package_raise("reconplogger", importer):
        import reconplogger
    return reconplogger


def setup_default_logger(data, level, caller):
    name = caller
    if isinstance(data, str):
        name = data
    elif isinstance(data, dict) and "name" in data:
        name = data["name"]
    logger = logging.getLogger(name)
    logger.parent = None
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    level = getattr(logging, level)
    for handler in logger.handlers:
        handler.setLevel(level)
    return logger


null_logger = logging.getLogger("jsonargparse_null_logger")


logging_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


def parse_logger(logger: Union[bool, str, dict, logging.Logger], caller):
    if not isinstance(logger, (bool, str, dict, logging.Logger)):
        raise ValueError(
            f"Expected logger to be an instance of (bool, str, dict, logging.Logger), but got {logger}."
        )
    if isinstance(logger, dict) and len(set(logger.keys()) - {"name", "level"}) > 0:
        value = {k: v for k, v in logger.items() if k not in {"name", "level"}}
        raise ValueError(f"Unexpected data to configure logger: {value}.")
    if logger is False:
        return null_logger
    level = "WARNING"
    if isinstance(logger, dict) and "level" in logger:
        level = logger["level"]
    if level not in logging_levels:
        raise ValueError(
            f"Got logger level {level!r} but must be one of {logging_levels}."
        )
    if (
        logger is True or (isinstance(logger, dict) and "name" not in logger)
    ) and reconplogger_support:
        kwargs = {"level": "DEBUG", "reload": True} if debug_mode_active() else {}
        logger = import_reconplogger("parse_logger").logger_setup(**kwargs)
    if not isinstance(logger, logging.Logger):
        logger = setup_default_logger(logger, level, caller)
    return logger


def debug_mode_active() -> bool:
    return os.getenv("JSONARGPARSE_DEBUG", "").lower() not in {"", "false", "no", "0"}


class LoggerProperty:
    """Class designed to be inherited by other classes to add a logger property."""

    def __init__(
        self, *args, logger: Union[bool, str, dict, logging.Logger] = False, **kwargs
    ):
        """Initializer for LoggerProperty class."""
        self.logger = logger  # type: ignore[assignment]
        super().__init__(*args, **kwargs)

    @property
    def logger(self) -> logging.Logger:
        """The logger property for the class.

        :getter: Returns the current logger.
        :setter: Sets the given logging.Logger as logger or sets the default logger
                 if given True/str(logger name)/dict(name, level), or disables logging
                 if given False.

        Raises:
            ValueError: If an invalid logger value is given.
        """
        return self._logger

    @logger.setter
    def logger(self, logger: Union[bool, str, dict, logging.Logger]):
        if logger is None:
            raise NotImplementedError('liberator did not resolve internal imports')
            from ._deprecated import deprecation_warning, logger_property_none_message

            deprecation_warning(
                (LoggerProperty.logger, None),
                logger_property_none_message,
                stacklevel=2,
            )
            logger = False
        if not logger and debug_mode_active():
            logger = {"level": "DEBUG"}
        self._logger = parse_logger(logger, type(self).__name__)


def ast_variable_load(name):
    return ast.Name(id=name, ctx=ast.Load())


ast_constant_attr = {ast.Constant: "value"}


class UnknownDefault:
    def __init__(self, resolver: str, data: Any = inspect._empty) -> None:
        self.resolver = resolver
        self.data = data

    def __repr__(self) -> str:
        value = f"{type(self).__name__.replace('Default', '')}<{self.resolver}>"
        if self.data != inspect._empty:
            value = f"{value} {self.data}"
        return value


def replace_generic_type_vars(params: ParamList, parent) -> None:
    if (
        is_generic_class(parent)
        and parent.__args__
        and getattr(parent.__origin__, "__parameters__", None)
    ):
        type_vars = dict(zip(parent.__origin__.__parameters__, parent.__args__))

        def replace_type_vars(annotation):
            if annotation in type_vars:
                return type_vars[annotation]
            if getattr(annotation, "__args__", None):
                origin = annotation.__origin__
                if (
                    sys.version_info < (3, 10)
                    and getattr(origin, "__module__", "") != "typing"
                ):
                    origin = getattr(
                        __import__("typing"), origin.__name__.capitalize(), origin
                    )
                return origin[tuple(replace_type_vars(a) for a in annotation.__args__)]
            return annotation

        for param in params:
            param.annotation = replace_type_vars(param.annotation)


parameter_attributes = [s[1:] for s in inspect.Parameter.__slots__]


def is_staticmethod(attr) -> bool:
    return isinstance(attr, staticmethod)


def is_property(attr) -> bool:
    return isinstance(attr, property)


def is_method(attr) -> bool:
    return (
        inspect.isfunction(attr)
        or attr.__class__.__name__ == "cython_function_or_method"
    ) and not is_staticmethod(attr)


def has_dunder_new_method(cls, attr_name):
    classes = inspect.getmro(get_generic_origin(cls))[1:]
    return (
        attr_name == "__init__"
        and cls.__new__ is not object.__new__
        and not any(cls.__new__ is c.__new__ for c in classes)
    )


def get_arg_kind_index(params, kind):
    return next((n for n, p in enumerate(params) if p.kind == kind), -1)


dict_ast = ast.dump(ast_variable_load("dict"))


current_mro: ContextVar = ContextVar("current_mro", default=(None, None))


def ast_get_call_positional_indexes(node):
    return [n for n, a in enumerate(node.args) if not isinstance(a, ast.Starred)]


def ast_get_call_keyword_names(node):
    return [kw_node.arg for kw_node in node.keywords if kw_node.arg]


ast_constant_types = tuple(ast_constant_attr.keys())


class ConditionalDefault(UnknownDefault):
    def __init__(self, resolver: str, data: Any) -> None:
        super().__init__(resolver, iter_to_set_str(data, sep=", "))


def unpack_typed_dict_kwargs(params: ParamList, kwargs_idx: int) -> int:
    kwargs = params[kwargs_idx]
    annotation = kwargs.annotation
    if is_unpack_typehint(annotation):
        params.pop(kwargs_idx)
        annotation_args: tuple = getattr(annotation, "__args__", tuple())
        assert len(annotation_args) == 1, "Unpack requires a single type argument"
        dict_annotations = annotation_args[0].__annotations__
        new_params = []
        for nm, annot in dict_annotations.items():
            new_params.append(
                ParamData(
                    name=nm,
                    annotation=annot,
                    default=inspect._empty,
                    kind=inspect._ParameterKind.KEYWORD_ONLY,
                    doc=None,
                    component=kwargs.component,
                    parent=kwargs.parent,
                    origin=kwargs.origin,
                )
            )
        # insert in-place
        assert kwargs_idx == len(params), "trailing params should yield a syntax error"
        params.extend(new_params)
        return -1
    return kwargs_idx


def split_args_and_kwargs(params: ParamList) -> Tuple[ParamList, ParamList]:
    args = [p for p in params if p.kind == kinds.POSITIONAL_ONLY]
    kwargs = [
        p for p in params if p.kind in {kinds.KEYWORD_ONLY, kinds.POSITIONAL_OR_KEYWORD}
    ]
    return args, kwargs


def replace_args_and_kwargs(
    params: ParamList, args: ParamList, kwargs: ParamList
) -> ParamList:
    args_idx = get_arg_kind_index(params, kinds.VAR_POSITIONAL)
    kwargs_idx = get_arg_kind_index(params, kinds.VAR_KEYWORD)
    if args_idx >= 0:
        params = params[:args_idx] + args + params[args_idx + 1 :]
        if kwargs_idx >= 0:
            kwargs_idx += len(args) - 1
    if kwargs_idx >= 0:
        existing_names = {
            p.name for p in params[:kwargs_idx] + params[kwargs_idx + 1 :]
        }
        kwargs = [p for p in kwargs if p.name not in existing_names]
        params = params[:kwargs_idx] + kwargs + params[kwargs_idx + 1 :]
    return params


def remove_given_parameters(node, params, removed_params: Optional[set] = None):
    given_args = set(ast_get_call_positional_indexes(node))
    given_kwargs = set(ast_get_call_keyword_names(node))
    input_params = params
    params = [p for n, p in enumerate(params) if n not in given_args]
    params = [p for p in params if p.name not in given_kwargs]
    if removed_params is not None and len(params) < len(input_params):
        removed_params.update(p.name for p in input_params if p.name in given_kwargs)
    return params


param_kwargs_pop_or_get = "**.pop|get():"


@contextmanager
def mro_context(parent):
    token = None
    if parent:
        classes, idx = current_mro.get()
        if not classes or classes[idx] is not parent:
            classes = [c for c in inspect.getmro(parent) if c is not object]
            token = current_mro.set((classes, 0))
    try:
        yield
    finally:
        if token:
            current_mro.reset(token)


kinds = inspect._ParameterKind


def is_param_subclass_instance_default(param: ParamData) -> bool:
    if is_dataclass_like(type(param.default)):
        return False
    raise NotImplementedError('liberator did not resolve internal imports')
    from ._typehints import ActionTypeHint, get_optional_arg, get_subclass_types

    annotation = get_optional_arg(param.annotation)
    class_types = get_subclass_types(annotation, callable_return=True)
    return bool(
        (class_types and isinstance(param.default, class_types))
        or (
            is_lambda(param.default)
            and ActionTypeHint.is_callable_typehint(annotation)
            and getattr(annotation, "__args__", None)
            and ActionTypeHint.is_subclass_typehint(
                annotation.__args__[-1], all_subtypes=False
            )
        )
    )


def is_method_or_property(attr) -> bool:
    return is_method(attr) or is_property(attr)


def is_lambda(value: Any) -> bool:
    return callable(value) and getattr(value, "__name__", "") == "<lambda>"


def is_init_field_pydantic2_dataclass(field) -> bool:
    from pydantic.fields import FieldInfo

    if isinstance(field.default, FieldInfo):
        # FieldInfo.init is new in pydantic 2.6
        return getattr(field.default, "init", None) is not False
    return field.init is not False


def is_init_field_attrs(field) -> bool:
    return field.init is not False


def is_classmethod(parent, component) -> bool:
    if parent:
        with suppress(AttributeError):
            return isinstance(
                inspect.getattr_static(parent, component.__name__), classmethod
            )
    return False


ignore_params = {"transformers.BertModel.from_pretrained": {"config_file_name"}}


def group_parameters(params_list: List[ParamList]) -> ParamList:
    if len(params_list) == 1:
        for param in params_list[0]:
            if not isinstance(param.origin, tuple):
                param.origin = None
        return params_list[0]
    grouped = []
    non_get_pop_count = 0
    params_dict = defaultdict(list)
    for params in params_list:
        if not (params[0].origin or "").startswith(param_kwargs_pop_or_get):  # type: ignore[union-attr]
            non_get_pop_count += 1
        for param in params:
            if param.kind != kinds.POSITIONAL_ONLY:
                params_dict[param.name].append(param)
    for params in params_dict.values():
        gparam = params[0]
        types = unique(
            p.annotation for p in params if p.annotation is not inspect._empty
        )
        defaults = unique(p.default for p in params if p.default is not inspect._empty)
        if len(params) >= non_get_pop_count and len(types) <= 1 and len(defaults) <= 1:
            gparam.origin = None
        else:
            gparam.parent = tuple(p.parent for p in params)
            gparam.component = tuple(p.component for p in params)
            gparam.origin = tuple(p.origin for p in params)
            if len(params) < non_get_pop_count:
                defaults += ["NOT_ACCEPTED"]
            gparam.default = ConditionalDefault("ast-resolver", defaults)
            if len(types) > 1:
                gparam.annotation = Union[tuple(types)] if types else inspect._empty
        docs = [p.doc for p in params if p.doc]
        gparam.doc = docs[0] if docs else None
        grouped.append(gparam)
    return grouped


def get_signature_parameters_and_indexes(component, parent, logger):
    signature_source = component
    if is_classmethod(parent, component):
        signature_source = component.__func__
    params = list(inspect.signature(signature_source).parameters.values())
    if parent:
        params = params[1:]
    args_idx = get_arg_kind_index(params, kinds.VAR_POSITIONAL)
    kwargs_idx = get_arg_kind_index(params, kinds.VAR_KEYWORD)
    doc_params = parse_docs(component, parent, logger)
    for num, param in enumerate(params):
        params[num] = ParamData(
            doc=doc_params.get(param.name),
            parent=parent,
            component=component,
            **{a: getattr(param, a) for a in parameter_attributes},
        )
    evaluate_postponed_annotations(params, signature_source, parent, logger)
    stubs = get_stub_types(params, signature_source, parent, logger)
    replace_generic_type_vars(params, parent)
    return params, args_idx, kwargs_idx, doc_params, stubs


def get_parameter_origins(component, parent) -> Optional[str]:
    raise NotImplementedError('liberator did not resolve internal imports')
    from ._typehints import get_subclass_types, sequence_origin_types

    if get_typehint_origin(component) in sequence_origin_types:
        component = get_subclass_types(component, also_lists=True)
    if isinstance(component, tuple):
        assert parent is None or len(component) == len(parent)
        return iter_to_set_str(
            get_parameter_origins(c, parent[n] if parent else None)
            for n, c in enumerate(component)
        )
    if parent:
        return f"{get_import_path(parent)}.{component.__name__}"
    return get_import_path(component)


def get_mro_parameters(method_name, get_parameters_fn, logger):
    classes, idx = current_mro.get()
    for num, cls in enumerate(classes[idx + 1 :], start=idx + 1):
        method = getattr(cls, method_name, None)
        remainder = classes[num + 1 :] + [object]
        if method and not any(
            method is getattr(c, method_name, None) for c in remainder
        ):
            current_mro.set((classes, num))
            return get_parameters_fn(cls, method, logger=logger)
    return []


def get_field_data_pydantic2_model(field, name, doc_params):
    default = field.default
    if field.is_required():
        default = inspect._empty
    elif field.default_factory:
        default = field.default_factory()

    return dict(
        annotation=field.rebuild_annotation(),
        default=default,
        doc=field.description or doc_params.get(name),
    )


def get_field_data_pydantic2_dataclass(field, name, doc_params):
    from pydantic.fields import FieldInfo
    from pydantic_core import PydanticUndefined

    default = inspect._empty
    # Identify the default.
    if isinstance(field.default, FieldInfo):
        # Pydantic 2 dataclasses stuff their FieldInfo into a
        # stdlib dataclasses.field's `default`; this is where the
        # actual default and default_factory live.
        if field.default.default is not PydanticUndefined:
            default = field.default.default
        elif field.default.default_factory is not PydanticUndefined:
            default = field.default.default_factory()
    elif field.default is not dataclasses.MISSING:
        default = field.default
    elif field.default_factory is not dataclasses.MISSING:
        default = field.default_factory()

    # Get the type, stripping Annotated like get_type_hints does.
    if is_annotated(field.type):
        field_type = get_annotated_base_type(field.type)
    else:
        field_type = field.type
    return dict(
        annotation=field_type,
        default=default,
        doc=doc_params.get(name),
    )


def get_field_data_pydantic1_model(field, name, doc_params):
    default = field.default
    if field.required:
        default = inspect._empty
    elif field.default_factory:
        default = field.default_factory()

    return dict(
        annotation=field.annotation,
        default=default,
        doc=field.field_info.description or doc_params.get(name),
    )


def get_field_data_attrs(field, name, doc_params):
    import attrs

    default = field.default
    if default is attrs.NOTHING:
        default = inspect._empty
    elif isinstance(default, attrs.Factory):
        default = default.factory()

    return dict(
        annotation=field.type,
        default=default,
        doc=doc_params.get(name),
    )


def get_component_and_parent(
    function_or_class: Union[Callable, Type],
    method_or_property: Optional[Union[str, Callable]] = None,
):
    if is_subclass(function_or_class, ClassFromFunctionBase) and method_or_property in {
        None,
        "__init__",
    }:
        function_or_class = function_or_class.wrapped_function  # type: ignore[union-attr]
        if isinstance(function_or_class, MethodType):
            method_or_property = function_or_class.__name__
            function_or_class = function_or_class.__self__  # type: ignore[assignment]
        else:
            method_or_property = None
    elif (
        inspect.isclass(get_generic_origin(function_or_class))
        and method_or_property is None
    ):
        method_or_property = "__init__"
    elif method_or_property and not isinstance(method_or_property, str):
        method_or_property = method_or_property.__name__
    parent = component = None
    if method_or_property:
        attr = inspect.getattr_static(
            get_generic_origin(function_or_class), method_or_property
        )
        if is_staticmethod(attr):
            component = getattr(function_or_class, method_or_property)
            return component, parent, method_or_property
        parent = function_or_class
        if has_dunder_new_method(function_or_class, method_or_property):
            component = getattr(function_or_class, "__new__")
        elif is_method(attr):
            component = attr
        elif is_property(attr):
            component = attr.fget
        elif isinstance(attr, classmethod):
            component = getattr(function_or_class, method_or_property)
        elif attr is not object.__init__:
            raise ValueError(
                f"Invalid or unsupported input: class={function_or_class}, method_or_property={method_or_property}"
            )
    else:
        if not callable(function_or_class):
            raise ValueError(f"Non-callable input: function={function_or_class}")
        component = function_or_class
    return component, parent, method_or_property


def ast_str(node):
    return getattr(ast, "unparse", ast.dump)(node)


ast_literals = {
    ast.dump(ast.parse(v, mode="eval").body): partial(ast.literal_eval, v)
    for v in ["{}", "[]"]
}


def ast_is_supported_super_call(node, self_name, log_debug) -> bool:
    supported = False
    args = node.func.value.args
    if not args and not node.func.value.keywords:
        supported = True
    elif (
        args
        and len(args) == 2
        and all(isinstance(a, ast.Name) for a in args)
        and self_name == args[1].id
        and not node.func.value.keywords
    ):
        classes, idx = current_mro.get()
        module = inspect.getmodule(classes[idx])
        for offset, cls in enumerate(classes[idx:]):
            if args[0].id == cls.__name__ and cls is getattr(
                module, cls.__name__, None
            ):
                current_mro.set((classes, idx + offset))
                supported = True
                break
    if not supported:
        log_debug(f"unsupported super parameters: {ast_str(node)}")
    return supported


def ast_is_super_call(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
    )


def ast_is_not(node) -> bool:
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)


def ast_is_kwargs_pop_or_get(node, value_dump) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and value_dump == ast.dump(node.func.value)
        and node.func.attr in {"pop", "get"}
        and len(node.args) == 2
        and isinstance(ast_get_constant_value(node.args[0]), str)
    )


def ast_is_dict_assign_with_value(node, value):
    if ast_is_dict_assign(node) and getattr(node.value, "keywords", None):
        value_dump = ast.dump(value)
        for keyword in [k.value for k in node.value.keywords]:
            if ast.dump(keyword) == value_dump:
                return True
    return False


def ast_is_dict_assign(node):
    return isinstance(node, ast_assign_type) and (
        isinstance(node.value, ast.Dict)
        or (isinstance(node.value, ast.Call) and ast.dump(node.value.func) == dict_ast)
    )


def ast_is_constant(node):
    return isinstance(node, ast_constant_types)


def ast_is_call_with_value(node, value_dump) -> bool:
    for argtype in ["args", "keywords"]:
        for arg in getattr(node, argtype):
            if (
                isinstance(getattr(arg, "value", None), ast.AST)
                and ast.dump(arg.value) == value_dump
            ):
                return True
    return False


def ast_is_attr_assign(node, container):
    for target in (
        ast_get_assign_targets(node) if isinstance(node, ast_assign_type) else []
    ):
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == container
        ):
            return target.attr
    return False


def ast_is_assign_with_value(node, value) -> bool:
    return isinstance(node, ast_assign_type) and ast.dump(node.value) == ast.dump(value)  # type: ignore[attr-defined]


def ast_get_name_and_attrs(node) -> List[str]:
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return names[::-1]


def ast_get_constant_value(node):
    assert ast_is_constant(node)
    return getattr(node, ast_constant_attr[node.__class__])


def ast_get_call_kwarg_with_value(node, value):
    value_dump = ast.dump(value)
    kwarg = None
    for arg in node.keywords:
        if (
            isinstance(getattr(arg, "value", None), ast.AST)
            and ast.dump(arg.value) == value_dump
        ):
            kwarg = arg
            break
    return kwarg


def ast_get_assign_targets(node):
    return node.targets if isinstance(node, ast.Assign) else [node.target]


def ast_attribute_load(container, name):
    return ast.Attribute(
        value=ast.Name(id=container, ctx=ast.Load()), attr=name, ctx=ast.Load()
    )


ast_assign_type: Tuple[(Type[ast.AST], ...)] = (ast.AnnAssign, ast.Assign)


def add_stub_types(
    stubs: Optional[Dict[str, Any]], params: ParamList, component
) -> None:
    if not stubs:
        return
    for param in params:
        if param.annotation == inspect._empty and param.name in stubs:
            param.annotation = stubs[param.name]
    known_params = {p.name for p in params}
    for name, stub in stubs.items():
        if name not in known_params:
            params.append(
                ParamData(
                    name=name,
                    annotation=stub,
                    default=UnknownDefault("stubs-resolver"),
                    kind=kinds.KEYWORD_ONLY,
                    component=component,
                )
            )


def get_parameters_from_pydantic_or_attrs(
    function_or_class: Union[Callable, Type],
    method_or_property: Optional[str],
    logger: logging.Logger,
) -> Optional[ParamList]:
    raise NotImplementedError('liberator did not resolve internal imports')
    from ._optionals import attrs_support, pydantic_support

    if method_or_property or not (pydantic_support or attrs_support):
        return None
    def yes(_):
        return True
    function_or_class = get_unaliased_type(function_or_class)
    fields_iterator = get_field_data = None
    if pydantic_support:
        pydantic_model = is_pydantic_model(function_or_class)
        if pydantic_model == 1:
            fields_iterator = function_or_class.__fields__.items()
            get_field_data = get_field_data_pydantic1_model
            is_init_field = yes
        elif pydantic_model > 1:
            fields_iterator = function_or_class.model_fields.items()
            get_field_data = get_field_data_pydantic2_model
            is_init_field = yes
        elif dataclasses.is_dataclass(function_or_class) and hasattr(
            function_or_class, "__pydantic_fields__"
        ):
            fields_iterator = dataclasses.fields(function_or_class)
            fields_iterator = {v.name: v for v in fields_iterator}.items()
            get_field_data = get_field_data_pydantic2_dataclass
            is_init_field = is_init_field_pydantic2_dataclass

    if not fields_iterator and attrs_support:
        import attrs

        if attrs.has(function_or_class):
            fields_iterator = {
                f.name: f for f in attrs.fields(function_or_class)
            }.items()
            get_field_data = get_field_data_attrs
            is_init_field = is_init_field_attrs

    if not fields_iterator or not get_field_data:
        return None

    params = []
    doc_params = parse_docs(function_or_class, None, logger)
    for name, field in fields_iterator:
        if is_init_field(field):
            params.append(
                ParamData(
                    name=name,
                    kind=kinds.KEYWORD_ONLY,
                    component=function_or_class,
                    **get_field_data(field, name, doc_params),
                )
            )
    evaluate_postponed_annotations(params, function_or_class, None, logger)
    return params


def get_parameters_by_assumptions(
    function_or_class: Union[Callable, Type],
    method_name: Optional[str] = None,
    logger: Union[bool, str, dict, logging.Logger] = True,
) -> ParamList:
    component, parent, method_name = get_component_and_parent(
        function_or_class, method_name
    )
    params, args_idx, kwargs_idx, _, stubs = get_signature_parameters_and_indexes(
        component, parent, logger
    )

    if parent and (args_idx >= 0 or kwargs_idx >= 0):
        with mro_context(parent):
            subparams = get_mro_parameters(
                method_name, get_parameters_by_assumptions, logger
            )
        if subparams:
            args, kwargs = split_args_and_kwargs(subparams)
            params = replace_args_and_kwargs(params, args, kwargs)

    params = replace_args_and_kwargs(params, [], [])
    add_stub_types(stubs, params, component)
    return params


class SourceNotAvailable(Exception):
    "Raised when the source code for some component is not available."


class ParametersVisitor(LoggerProperty, ast.NodeVisitor):
    def __init__(
        self,
        function_or_class: Union[Callable, Type],
        method_or_property: Optional[Union[str, Callable]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.component, self.parent, _ = get_component_and_parent(
            function_or_class, method_or_property
        )

    def log_debug(self, message) -> None:
        self.logger.debug(f"AST resolver: {message}")

    def parse_source_tree(self):
        """Parses the component's AST and sets the component and parent nodes."""
        if hasattr(self, "component_node"):
            return
        try:
            source = textwrap.dedent(inspect.getsource(self.component))
            tree = ast.parse(source)
            assert isinstance(tree, ast.Module) and len(tree.body) == 1
            self.component_node = tree.body[0]
            self.self_name = (
                self.component_node.args.args[0].arg if self.parent else None
            )
        except Exception as ex:
            raise SourceNotAvailable(
                f"Problems getting source code for {self.component}: {ex}"
            ) from ex

    def visit_Assign(self, node):
        do_generic_visit = True
        for key, value in self.find_values.items():
            if ast_is_assign_with_value(node, value):
                self.add_value(key, node)
                do_generic_visit = False
                break
            elif ast_is_dict_assign_with_value(node, value):
                self.add_value(key, node)
                do_generic_visit = False
        if do_generic_visit:
            if ast_is_dict_assign(node):
                for target in [deepcopy(t) for t in ast_get_assign_targets(node)]:
                    target.ctx = ast.Load()
                    self.dict_assigns[ast.dump(target)] = node
            else:
                self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit_Assign(node)

    def visit_Call(self, node):
        for key, value in self.find_values.items():
            value_dump = ast.dump(value)
            if ast_is_call_with_value(node, value_dump):
                if isinstance(node.func, ast.Attribute):
                    value_dump = ast.dump(node.func.value)
                    if value_dump in self.dict_assigns:
                        self.add_value(key, self.dict_assigns[value_dump])
                        continue
                self.add_value(key, node)
            elif ast_is_kwargs_pop_or_get(node, value_dump):
                self.add_value(key, node)
        self.generic_visit(node)

    def visit_If(self, node):
        is_test_not = ast_is_not(node.test)
        test_node = node.test.operand if is_test_not else node.test
        component_globals = self.get_component_globals()
        if isinstance(test_node, ast.Name) and test_node.id in component_globals:
            condition = bool(component_globals[test_node.id])
            if is_test_not:
                condition = not condition
            body = node.body if condition else node.orelse
            node = ast.If(test=ast.Constant(value=True), body=body, orelse=[])
        self.generic_visit(node)

    def visit_Import(self, node: Union[ast.Import, ast.ImportFrom]) -> None:
        for alias in node.names:
            self.import_names[alias.asname or alias.name] = node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.visit_Import(node)

    def add_value(self, key, node):
        source = None
        if isinstance(node, ast.Call):
            name = False
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                name = node.func.value.id
            if name and name in self.import_names:
                source = self.import_names[name]
        self.values_found.append((key, node, source))

    def find_values_usage(self, values):
        self.find_values = values
        self.values_found = []
        self.dict_assigns = {}
        self.import_names = {}
        self.visit(self.component_node)
        return self.values_found

    def get_component_globals(self):
        return vars(import_module(self.component.__module__))

    def get_component_from_source(self, name, source):
        aliases = {}
        ast_exec = ast.parse("")
        ast_exec.body = [source]
        try:
            exec(compile(ast_exec, filename="<ast>", mode="exec"), aliases, aliases)
        except Exception as ex:
            if self.logger:
                self.logger.debug(
                    f"Failed to get '{name}' from '{ast_str(source)}'", exc_info=ex
                )
        return aliases.get(name)

    def get_node_component(self, node, source) -> Optional[Tuple[Type, Optional[str]]]:
        function_or_class = method_or_property = None
        module = inspect.getmodule(self.component)
        if isinstance(node.func, ast.Name):
            if (
                is_classmethod(self.parent, self.component)
                and node.func.id == self.self_name
            ):
                function_or_class = self.parent
            elif hasattr(module, node.func.id):
                function_or_class = getattr(module, node.func.id)
            elif source:
                function_or_class = self.get_component_from_source(node.func.id, source)
        elif isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            if self.parent and ast.dump(node.func.value) == ast.dump(
                ast_variable_load(self.self_name)
            ):
                function_or_class = self.parent
                method_or_property = node.func.attr
            else:
                container = None
                if hasattr(module, node.func.value.id):
                    container = getattr(module, node.func.value.id)
                elif source:
                    container = self.get_component_from_source(
                        node.func.value.id, source
                    )
                if inspect.isclass(container):
                    function_or_class = container
                    method_or_property = node.func.attr
                elif hasattr(container, node.func.attr):
                    function_or_class = getattr(container, node.func.attr)
        if not function_or_class:
            self.log_debug(f"not supported: {ast_str(node)}")
            return None
        return function_or_class, method_or_property

    def match_call_that_uses_attr(self, node, source, attr_name):
        params = None
        if isinstance(node, ast.Call):
            params = []
            value = ast_attribute_load(self.self_name, attr_name)
            kwarg = ast_get_call_kwarg_with_value(node, value)
            if kwarg:
                if kwarg.arg:
                    self.log_debug(
                        f"kwargs attribute given as keyword parameter not supported: {ast_str(node)}"
                    )
                else:
                    get_param_args = self.get_node_component(node, source)
                    if get_param_args:
                        try:
                            params = get_signature_parameters(
                                *get_param_args, logger=self.logger
                            )
                        except Exception:
                            self.log_debug(
                                f"failed to get parameters for call that uses attr: {get_param_args}"
                            )
            params = remove_given_parameters(node, params)
        return params

    def replace_param_default_subclass_specs(self, params: List[ParamData]) -> None:
        params = [p for p in params if is_param_subclass_instance_default(p)]
        if params:
            self.parse_source_tree()
            default_nodes = self.get_default_nodes({p.name for p in params})
            assert len(params) == len(default_nodes)
            raise NotImplementedError('liberator did not resolve internal imports')
            from ._typehints import get_subclass_types

            for param, default_node in zip(params, default_nodes):
                lambda_default = is_lambda(param.default)
                node = default_node
                num_positionals = 0
                if lambda_default:
                    node = default_node.body
                    num_positionals = len(param.annotation.__args__) - 1
                class_type = self.get_call_class_type(node)
                subclass_types = get_subclass_types(
                    param.annotation, callable_return=True
                )
                if not (
                    class_type
                    and subclass_types
                    and is_subclass(class_type, subclass_types)
                ):
                    continue
                subclass_spec: dict = dict(
                    class_path=get_import_path(class_type), init_args=dict()
                )
                for kwarg in node.keywords:
                    if kwarg.arg and ast_is_constant(kwarg.value):
                        subclass_spec["init_args"][kwarg.arg] = ast_get_constant_value(
                            kwarg.value
                        )
                    else:
                        subclass_spec.clear()
                        break
                if not subclass_spec or len(node.args) - num_positionals > 0:
                    self.log_debug(
                        f"unsupported class instance default: {ast_str(default_node)}"
                    )
                elif subclass_spec:
                    if not subclass_spec["init_args"]:
                        del subclass_spec["init_args"]
                    param.default = subclass_spec

    def get_call_class_type(self, node) -> Optional[type]:
        names = ast_get_name_and_attrs(getattr(node, "func", None))
        class_type = self.get_component_globals().get(names[0]) if names else None
        for name in names[1:]:
            class_type = getattr(class_type, name, None)
        return class_type if inspect.isclass(class_type) else None

    def get_default_nodes(self, param_names: set):
        node = self.component_node.args
        arg_nodes = getattr(node, "posonlyargs", []) + node.args
        default_nodes = [None] * (len(arg_nodes) - len(node.defaults)) + node.defaults
        default_nodes = [
            d for n, d in enumerate(default_nodes) if arg_nodes[n].arg in param_names
        ]
        return default_nodes

    def get_kwargs_pop_or_get_parameter(self, node, component, parent, doc_params):
        name = ast_get_constant_value(node.args[0])
        if ast_is_constant(node.args[1]):
            default = ast_get_constant_value(node.args[1])
        else:
            default = ast.dump(node.args[1])
            if default in ast_literals:
                default = ast_literals[default]()
            else:
                default = UnknownDefault("ast-resolver")
                self.log_debug(f"unsupported kwargs pop/get default: {ast_str(node)}")
        return ParamData(
            name=name,
            annotation=inspect._empty,
            default=default,
            kind=kinds.KEYWORD_ONLY,
            doc=doc_params.get(name),
            parent=parent,
            component=component,
            origin=param_kwargs_pop_or_get + self.get_node_origin(node),
        )

    def get_parameters_args_and_kwargs(self) -> Tuple[ParamList, ParamList]:
        self.parse_source_tree()
        args_name = getattr(self.component_node.args.vararg, "arg", None)
        kwargs_name = getattr(self.component_node.args.kwarg, "arg", None)
        values_to_find = {}
        if args_name:
            values_to_find[args_name] = ast_variable_load(args_name)
        if kwargs_name:
            values_to_find[kwargs_name] = ast_variable_load(kwargs_name)

        values_found = self.find_values_usage(values_to_find)
        if not values_found:
            return [], []

        params_list = []
        removed_params: Set[str] = set()
        kwargs_value = kwargs_name and values_to_find[kwargs_name]
        kwargs_value_dump = kwargs_value and ast.dump(kwargs_value)
        for node, source in [(v, s) for k, v, s in values_found if k == kwargs_name]:
            if isinstance(node, ast.Call):
                if ast_is_kwargs_pop_or_get(node, kwargs_value_dump):
                    param = self.get_kwargs_pop_or_get_parameter(
                        node, self.component, self.parent, self.doc_params
                    )
                    params_list.append([param])
                    continue
                kwarg = ast_get_call_kwarg_with_value(node, kwargs_value)
                params = []
                if kwarg.arg:
                    self.log_debug(
                        f"kwargs given as keyword parameter not supported: {ast_str(node)}"
                    )
                elif self.parent and ast_is_super_call(node):
                    if ast_is_supported_super_call(
                        node, self.self_name, self.log_debug
                    ):
                        params = get_mro_parameters(
                            node.func.attr,  # type: ignore[attr-defined]
                            get_signature_parameters,
                            self.logger,
                        )
                else:
                    get_param_args = self.get_node_component(node, source)
                    if get_param_args:
                        params = get_signature_parameters(
                            *get_param_args, logger=self.logger
                        )
                params = remove_given_parameters(node, params, removed_params)
                if params:
                    self.add_node_origins(params, node)
                    params_list.append(params)
            elif isinstance(node, ast_assign_type):
                self_attr = self.parent and ast_is_attr_assign(node, self.self_name)
                if self_attr:
                    params = self.get_parameters_attr_use_in_members(self_attr)
                    if params:
                        self.add_node_origins(params, node)
                        params_list.append(params)
                else:
                    self.log_debug(f"unsupported type of assign: {ast_str(node)}")

        params = group_parameters(params_list)
        params = [p for p in params if p.name not in removed_params]
        return split_args_and_kwargs(params)

    def get_parameters_attr_use_in_members(self, attr_name) -> ParamList:
        attr_value = ast_attribute_load(self.self_name, attr_name)
        member_names = [
            name
            for name, _ in inspect.getmembers(self.parent)
            if not name.startswith("__")
            and is_method_or_property(inspect.getattr_static(self.parent, name))
        ]
        for member_name in member_names:
            assert self.parent is not None
            visitor = ParametersVisitor(self.parent, member_name, logger=self.logger)
            kwargs = visitor.get_parameters_call_attr(attr_name, attr_value)
            if kwargs is not None:
                return kwargs
        self.log_debug(
            f"did not find use of {self.self_name}.{attr_name} in members of {self.parent}"
        )
        return []

    def get_node_origin(self, node) -> str:
        return f"{get_parameter_origins(self.component, self.parent)}:{node.lineno}"

    def add_node_origins(self, params: ParamList, node) -> None:
        origin = None
        for param in params:
            if param.origin is None:
                if not origin:
                    origin = self.get_node_origin(node)
                param.origin = origin

    def get_parameters_call_attr(
        self, attr_name: str, attr_value: ast.AST
    ) -> Optional[ParamList]:
        self.parse_source_tree()
        values_to_find = {attr_name: attr_value}
        values_found = self.find_values_usage(values_to_find)
        matched = []
        if values_found:
            for _, node, source in values_found:
                match = self.match_call_that_uses_attr(node, source, attr_name)
                if match:
                    self.add_node_origins(match, node)
                    matched.append(match)
            matched = group_parameters(matched)
        return matched or None

    def remove_ignore_parameters(self, params: ParamList) -> ParamList:
        import_path = get_import_path(self.component)
        if import_path in ignore_params:
            params = [p for p in params if p.name not in ignore_params[import_path]]
        return params

    def get_parameters(self) -> ParamList:
        if self.component is None:
            return []
        params, args_idx, kwargs_idx, doc_params, stubs = (
            get_signature_parameters_and_indexes(
                self.component, self.parent, self.logger
            )
        )
        self.replace_param_default_subclass_specs(params)
        if kwargs_idx >= 0:
            kwargs_idx = unpack_typed_dict_kwargs(params, kwargs_idx)
        if args_idx >= 0 or kwargs_idx >= 0:
            self.doc_params = doc_params
            with mro_context(self.parent):
                args, kwargs = self.get_parameters_args_and_kwargs()
            params = replace_args_and_kwargs(params, args, kwargs)
        add_stub_types(stubs, params, self.component)
        params = self.remove_ignore_parameters(params)
        return params


def get_signature_parameters(
    function_or_class: Union[Callable, Type],
    method_or_property: Optional[str] = None,
    logger: Union[bool, str, dict, logging.Logger] = True,
) -> ParamList:
    """Get parameters by inspecting ASTs or by inheritance assumptions if source not available.

    In contrast to inspect.signature, it follows the use of *args and **kwargs
    attempting to find all accepted named parameters.

    Args:
        function_or_class: The callable object from which to get the signature
            parameters.
        method_or_property: For classes, the name of the method or property from
            which to get the signature parameters. If not provided it returns
            the parameters for ``__init__``.
        logger: Useful for debugging. Only logs at ``DEBUG`` level.

    Example:
        >>> # xdoctest: +SKIP("not ready")
        >>> from scriptconfig.introspection.complex_introspection import *  # NOQA
        >>> function_or_class = get_signature_parameters
        >>> params = get_signature_parameters(function_or_class)
        >>> import ubelt as ub
        >>> print(f'params = {ub.urepr(params, nl=1)}')
    """
    logger = parse_logger(logger, "get_signature_parameters")
    try:
        params = get_parameters_from_pydantic_or_attrs(
            function_or_class, method_or_property, logger
        )
        if params is not None:
            return params
        visitor = ParametersVisitor(
            function_or_class, method_or_property, logger=logger
        )
        return visitor.get_parameters()
    except Exception as ex:
        cause = "Source not available"
        exc_info = None
        if not isinstance(ex, SourceNotAvailable):
            cause = "Problems with AST resolving"
            exc_info = ex
        logger.debug(
            f"{cause}, falling back to parameters by assumptions: function_or_class={function_or_class} "
            f"method_or_property={method_or_property}: {ex}",
            exc_info=exc_info,
        )
        return get_parameters_by_assumptions(
            function_or_class, method_or_property, logger
        )
