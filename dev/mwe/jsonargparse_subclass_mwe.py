import jsonargparse
import inspect
from jsonargparse.parameter_resolvers import ParamData
from jsonargparse.signatures import get_signature_parameters
from jsonargparse.signatures import get_doc_short_description


class MyArgumentParser(jsonargparse.ArgumentParser):
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
        ## Create group if requested ##
        doc_group = get_doc_short_description(function_or_class, method_name, self.logger)
        component = getattr(function_or_class, method_name) if method_name else function_or_class
        group = self._create_group_if_requested(component, nested_key, as_group, doc_group, instantiate=instantiate)

        params = get_signature_parameters(function_or_class, method_name, logger=self.logger)

        if hasattr(function_or_class, '__custom__'):
            # Hack to insert our method for explicit parameterization
            __custom__ = function_or_class.__custom__
            for key, info in __custom__.items():
                type = info.get('type', None)
                if type is None or not isinstance(type, type):
                    annotation = inspect._empty
                else:
                    annotation = type
                param = ParamData(
                    name=key,
                    annotation=annotation,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    default=info.get('value', None),
                    doc=info.get('help', None),
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
        # import ubelt as ub
        # print('added_args = {}'.format(ub.repr2(added_args, nl=1)))
        return added_args


class ClassA:
    def __init__(self, a=None):
        self.a = a


class ClassB(ClassA):
    __custom__ = {
        'b_name': {'value': [1, 2, 3], 'type': str, help: 'hello world'}
    }
    def __init__(self, b=3, **kwargs):
        self.b = b
        self.__dict__.update(kwargs)


class ClassC:
    __custom__ = {
        'c_name': {'type': str, help: 'hello world'}
    }
    def __init__(self, c=None, **kwargs):
        self.c = c
        self.__dict__.update(kwargs)


# Monkey patch jsonargparse so its subcommands use our extended functionality
MONKEY_PATCH = 1
if MONKEY_PATCH:
    jsonargparse.ArgumentParser = MyArgumentParser
    jsonargparse.core.ArgumentParser = MyArgumentParser


def main():
    parser = MyArgumentParser()
    parser.add_subclass_arguments(ClassA, nested_key='class_a', fail_untyped=False)
    parser.add_class_arguments(ClassC, nested_key='class_c', fail_untyped=False, sub_configs=True)
    config = parser.parse_args()
    instances = parser.instantiate_classes(config)
    if hasattr(instances, 'class_a'):
        print(f'{instances.class_a.__dict__=}')
    print(f'{instances.class_c.__dict__=}')


if __name__ == '__main__':
    """
    Ignore:
        pip install jsonargparse==4.14.1
        pip install jsonargparse -U

    CommandLine:
        python jsonargparse_subclass_mwe.py --help
        python jsonargparse_subclass_mwe.py --class_a.help ClassB
        python jsonargparse_subclass_mwe.py --class_a=ClassB --class_a.init_args.b_name=foo
        python jsonargparse_subclass_mwe.py --class_c.c_name=bar
        python jsonargparse_subclass_mwe.py
        --class_a=ClassA
    """
    main()
