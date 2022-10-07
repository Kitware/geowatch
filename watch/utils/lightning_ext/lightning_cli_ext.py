"""
This module is an exension of jsonargparse and lightning CLI that will respect
scriptconfig style arguments
"""
import ubelt as ub
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from typing import Callable, List, Type, Union
from jsonargparse import class_from_function
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class LightningArgumentParser_Extension(LightningArgumentParser):
    """


    Refactor references:
        ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/pytorch_lightning/cli.py
        ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/jsonargparse/core.py
        ~/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/jsonargparse/signatures.py


    """

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
            print(f'Parse scriptconfig params for: function_or_class={function_or_class}')
            # Specify our own set of explicit parameters here
            # pretend like things in scriptconfig are from the signature
            from jsonargparse.parameter_resolvers import ParamData
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
                # print(f'add scriptconfig param={param}')
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
        # import ubelt as ub
        # print('added_args = {}'.format(ub.repr2(added_args, nl=1)))
        return added_args

    def add_lightning_class_args(
        self,
        lightning_class: Union[
            Callable[..., Union[Trainer, LightningModule, LightningDataModule, Callback]],
            Type[Trainer],
            Type[LightningModule],
            Type[LightningDataModule],
            Type[Callback],
        ],
        nested_key: str,
        subclass_mode: bool = False,
        required: bool = True,
    ) -> List[str]:
        """Adds arguments from a lightning class to a nested key of the parser.

        Args:
            lightning_class: A callable or any subclass of {Trainer, LightningModule, LightningDataModule, Callback}.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
            required: Whether the argument group is required.

        Returns:
            A list with the names of the class arguments added.
        """
        print(f'[Lightning.CLI] Add args for lightning_class={lightning_class}')
        if callable(lightning_class) and not isinstance(lightning_class, type):
            lightning_class = class_from_function(lightning_class)

        if isinstance(lightning_class, type) and issubclass(
            lightning_class, (Trainer, LightningModule, LightningDataModule, Callback)
        ):
            if issubclass(lightning_class, Callback):
                self.callback_keys.append(nested_key)

            # Try 2: Revert to original behavior
            instantiate = not issubclass(lightning_class, Trainer)
            # Try 1:
            # Our extension will defer how the model is instantiated
            # because we need the dataset to be setup before we create it.
            # instantiate = not issubclass(lightning_class, (Trainer, LightningModule))

            if subclass_mode:
                return self.add_subclass_arguments(
                    lightning_class, nested_key,
                    fail_untyped=False,
                    required=required,
                    # instantiate=instantiate
                )

            return self.add_class_arguments(
                lightning_class,
                nested_key,
                fail_untyped=False,
                instantiate=instantiate,
                sub_configs=True,
            )
        raise MisconfigurationException(
            f"Cannot add arguments from: {lightning_class}. You should provide either a callable or a subclass of: "
            "Trainer, LightningModule, LightningDataModule, or Callback."
        )


class LightningCLI_Extension(LightningCLI):

    def init_parser(self, **kwargs):
        # Hack in our modified parser
        DEBUG = 0
        if DEBUG:
            kwargs['error_handler'] = None
        return LightningArgumentParser_Extension(**kwargs)

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes."""

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        # datavars = self.config.fit.data.__dict__
        # modelvars = self.config.fit.model.__dict__
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")

        ### Try 1
        # Call datamodule setup before instantiate
        # TODO: only need to do this IF these aren't specified on the CLI,
        # which could be a useful optimization (but also a shoegun).
        # self.datamodule.setup('fit')
        # self.config_init.fit.model.input_sensorchan = self.datamodule.input_sensorchan
        # self.config_init.fit.model.classes = self.datamodule.classes

        # Instantiate the model ourselves
        # modelkw = self._get(self.config_init, "model")
        # modelcls = self.model_class
        # self.model = modelcls(**modelkw)

        ### Try 2: Use unmodified instantiate_classes and put all the hacks
        ### in the link_arguments call.
        self.model = self._get(self.config_init, "model")

        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()
