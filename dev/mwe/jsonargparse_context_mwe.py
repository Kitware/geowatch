import jsonargparse


class ClassB:
    def __init__(self, a=None, b=3, b_name=3, **kwargs):
        self.b = b
        self.b_name = b_name
        self.__dict__.update(kwargs)


class ClassC:
    def __init__(self, c=None, c_name=None, **kwargs):
        self.c = c
        self.c_name = c_name
        self.__dict__.update(kwargs)


def main():
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(ClassB, nested_key='class_b', fail_untyped=False)
    parser.add_class_arguments(ClassC, nested_key='class_c', fail_untyped=False)

    parser.link_arguments(
        "class_b.b_name",
        "class_b.b_name",
        compute_fn=lambda x: str(x).upper(),
        apply_on='instantiate',
    )
    config = parser.parse_args()
    instances = parser.instantiate_classes(config)
    print(f'{instances.class_b.__dict__=}')
    print(f'{instances.class_c.__dict__=}')


if __name__ == '__main__':
    """
    Ignore:
        pip install jsonargparse==4.14.1
        pip install jsonargparse -U

    CommandLine:
        python jsonargparse_context_mwe.py --help
        python jsonargparse_context_mwe.py \
            --class_b.b_name=foo \
            --class_b.a=2 \
            --class_c.c=3
    """
    main()
