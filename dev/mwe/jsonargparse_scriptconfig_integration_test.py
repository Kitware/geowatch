import scriptconfig as scfg


class MyClassConfig(scfg.DataConfig):
    key1 = scfg.Value(1, alias=['key_one'], help='description1')
    key2 = scfg.Value(None, type=str, help='description1')
    key3 = scfg.Value(False, isflag=True, help='description1')
    key4 = 123
    key5 = '123'


class MyClass:
    __scriptconfig__ = MyClassConfig

    def __init__(self, regular_param1, **kwargs):
        self.regular_param1 = regular_param1
        self.config = MyClassConfig(**kwargs)


def main():
    import jsonargparse
    import shlex
    import ubelt as ub
    import rich
    from rich.markup import escape
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(MyClass, nested_key='my_class', fail_untyped=False, sub_configs=True)
    parser.add_argument('--foo', default='bar')
    parser.add_argument('-b', '--baz', '--buzz', default='bar')
    print('Parse Args')
    cases = [
        '',
        '--my_class.key1 123',
        '--my_class.key_one 123ab',
        '--my_class.key4 strings-are-ok',
    ]
    for case_idx, case in enumerate(cases):
        print('--- Case {case_idx} ---')
        print(f'case={case}')
        args = shlex.split(case)
        config = parser.parse_args(args)
        instances = parser.instantiate_classes(config)

        my_class = instances.my_class
        rich.print(f'config = {escape(ub.urepr(config, nl=2))}')
        rich.print(f'my_class.config = {escape(ub.urepr(my_class.config, nl=2))}')
        print('---')


if __name__ == '__main__':
    """
    CommandLine:
        cd ~/code/geowatch/dev/mwe/
        python jsonargparse_scriptconfig_integration_test.py
    """
    main()
