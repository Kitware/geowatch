"""
This module is an exension of jsonargparse and lightning CLI that will respect
scriptconfig style arguments
"""
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.cli import LightningArgumentParser


class LightningArgumentParser_Extension(LightningArgumentParser):

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
        from jsonargparse.signatures import get_signature_parameters
        from jsonargparse.signatures import get_doc_short_description

        ## Create group if requested ##
        doc_group = get_doc_short_description(function_or_class, method_name, self.logger)
        component = getattr(function_or_class, method_name) if method_name else function_or_class
        group = self._create_group_if_requested(component, nested_key, as_group, doc_group, instantiate=instantiate)

        params = get_signature_parameters(function_or_class, method_name, logger=self.logger)

        if hasattr(function_or_class, '__scriptconfig__'):
            # Specify our own set of explicit parameters here
            # pretend like things in scriptconfig are from the signature
            from jsonargparse.parameter_resolvers import ParamData
            import inspect
            # Hack to insert our method for explicit parameterization
            config_cls = function_or_class.__scriptconfig__
            config_cls.default.keys()

            for key, value in config_cls.default.items():
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
                params.append(param)

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

        return added_args


class LightningCLI_Extension(LightningCLI):

    def init_parser(self, **kwargs):
        # Hack in our modified parser
        return LightningArgumentParser_Extension(**kwargs)
