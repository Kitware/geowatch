import jsonargparse
import pathlib


class Data:
    def __init__(self, fpath=None):
        self.fpath = pathlib.Path(fpath)
        self.content = None

    def setup(self):
        self.content = self.fpath.read_text()
        return self


class Model:
    def __init__(self, content=None, param1=None):
        assert content is not None
        self.content = content


def main():
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(Data, nested_key='data', fail_untyped=False)
    parser.add_class_arguments(Model, nested_key='model', fail_untyped=False)

    parser.link_arguments('data.content', 'model.content', apply_on='instantiate')

    foo_fpath = pathlib.Path('foo.txt')
    foo_fpath.write_text('content of a file')

    config = parser.parse_args(args=['--data.fpath', str(foo_fpath)])
    instances = parser.instantiate_classes(config)


if __name__ == '__main__':
    main()


def main_v2():
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(Data, nested_key='data', fail_untyped=False)
    parser.add_class_arguments(Model, nested_key='model', fail_untyped=False)

    parser.link_arguments('data', 'model.content', apply_on='instantiate', compute_fn=lambda data: data.setup().content)

    foo_fpath = pathlib.Path('foo.txt')
    foo_fpath.write_text('content of a file')

    config = parser.parse_args(args=['--data.fpath', str(foo_fpath)])

    instances = parser.instantiate_classes(config)
    model = instances.model
    print(f'model.content={model.content}')


def main_ideal():
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(Data, nested_key='data', fail_untyped=False)
    parser.add_class_arguments(Model, nested_key='model', fail_untyped=False)

    # Proposed new feature
    parser.add_after_instantiate('data', lambda obj: obj.setup())

    parser.link_arguments('data', 'model.content', apply_on='instantiate')

    foo_fpath = pathlib.Path('foo.txt')
    foo_fpath.write_text('content of a file')

    config = parser.parse_args(args=['--data.fpath', str(foo_fpath)])
    instances = parser.instantiate_classes(config)
