from torch import nn
import torch
import os
import math
import ubelt as ub
import kwimage

millnames = ['', ' K', ' M', ' B', ' T']


try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


def millify(n):
    n = float(n)
    millidx = max(0, min(len(millnames) - 1,
                         int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    return '{:.2f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


def _map_location(storage, location):
    """
    Helper when calling `torch.load` to keep the data on the CPU

    Args:
        storage (torch.Storage) : the initial deserialization of the
            storage of the data read by `torch.load`, residing on the CPU.
        location (str): tag identifiying the location the data being read
            by `torch.load` was originally saved from.

    Returns:
        torch.Storage : the storage
    """
    return storage


def load_model_from_package(package_path):
    """
    Loads a kitware-flavor torch package (requires a package_header exists)

    Notes:
        * I don't like that we need to know module_name and arch_name a-priori
          given a path to a package, I just want to be able to construct
          the model instance. The package header solves this.

    Ignore:
        >>> from geowatch.tasks.fusion.utils import *  # NOQA
        >>> import geowatch
        >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_expt')
        >>> package_path = dvc_dpath / 'models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt'
        >>> model = load_model_from_package(package_path)
    """
    from geowatch.monkey import monkey_torchmetrics
    from geowatch.monkey import monkey_kwcoco
    monkey_torchmetrics.fix_torchmetrics_compatability()
    monkey_kwcoco.fix_sorted_set()
    from torch import package
    import json
    # imp = package.PackageImporter(package_path)
    import pathlib
    if not isinstance(package_path, (str, pathlib.Path)):
        raise TypeError(type(package_path))

    package_path = os.fspath(package_path)

    try:
        imp = package.PackageImporter(package_path)
    except (RuntimeError, ImportError):
        import warnings
        warnings.warn(
            f'Failed to import package {package_path} with normal machanism. '
            'Falling back to hacked mechanim')
        imp = _try_fixed_package_import(package_path)
    # Assume this standardized header information exists that tells us the
    # name of the resource corresponding to the model
    try:
        package_header = json.loads(imp.load_text(
            'package_header', 'package_header.json'))
    except Exception:
        print('warning: no standard package header')
        try:
            package_header = json.loads(imp.load_text(
                'kitware_package_header', 'kitware_package_header.json'))
        except Exception:
            package_header = imp.load_pickle(
                'kitware_package_header', 'kitware_package_header.pkl')
        print('warning: old package header?')
    arch_name = package_header['arch_name']
    module_name = package_header['module_name']

    model = imp.load_pickle(module_name, arch_name, map_location=_map_location)

    if 0:
        imp.file_structure()['package_header']

    # Add extra metadata to the model
    # raise Exception("foo")
    config_candidates = {
        "config_cli_yaml": "config.yaml",
        "fit_config": "fit_config.yaml",
    }
    for candidate_dest, candidate_fpath in config_candidates.items():
        try:
            fit_config_text = imp.load_text('package_header', candidate_fpath)
        except Exception:
            print(f"[load_model_from_package] Warning: did not find {candidate_dest} at {candidate_fpath}")
            pass
        else:
            import io
            import yaml
            file = io.StringIO(fit_config_text)
            # Note: types might be wrong here
            fit_config = yaml.safe_load(file)
            # model.fit_config = fit_config
            setattr(model, candidate_dest, fit_config)

    model.package_path = package_path
    return model


def load_model_header(package_path):
    """
    Only grabs header info from a packaged model.

    Ignore:
        >>> from geowatch.tasks.fusion.utils import *  # NOQA
        >>> import geowatch
        >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_expt')
        >>> package_path = dvc_dpath / 'models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt'
        >>> model = load_model_header(package_path)
    """
    import zipfile
    zfile = zipfile.ZipFile(package_path)
    names = zfile.namelist()
    relevant_names = {}
    for name in names:
        if name.endswith('package_header/package_header.json'):
            relevant_names['header'] = name
        if name.endswith('package_header/config.yaml'):
            relevant_names['config'] = name

    from kwutil.util_yaml import Yaml
    relevant_data = {}
    for key, name in relevant_names.items():
        text = zfile.read(name).decode()
        data = Yaml.coerce(text)
        relevant_data[key] = data
    return relevant_data


def _try_fixed_package_import(package_path):
    from torch import package
    import torch
    from typing import Any
    from torch.package.package_importer import (
        DirectoryReader, _PackageNode, PackageMangler, PackageUnpickler,
        _ModuleNode, _ExternNode)
    import builtins
    import os
    from pathlib import Path

    class CustomPackageImporter(package.PackageImporter):
        def __init__(self, file_or_buffer, module_allowed=lambda module_name: True):
            torch._C._log_api_usage_once("torch.package.PackageImporter")

            self.zip_reader: Any
            if isinstance(file_or_buffer, torch._C.PyTorchFileReader):
                self.filename = "<pytorch_file_reader>"
                self.zip_reader = file_or_buffer
            elif isinstance(file_or_buffer, (Path, str)):
                self.filename = str(file_or_buffer)
                if not os.path.isdir(self.filename):
                    self.zip_reader = torch._C.PyTorchFileReader(self.filename)
                else:
                    self.zip_reader = DirectoryReader(self.filename)
            else:
                self.filename = "<binary>"
                self.zip_reader = torch._C.PyTorchFileReader(file_or_buffer)

            self.root = _PackageNode(None)
            self.modules = {}
            self.extern_modules = self._read_extern()

            self.errors = []

            for extern_module in self.extern_modules:
                if not module_allowed(extern_module):
                    self.errors.append((
                        f"package '{file_or_buffer}' needs the external module '{extern_module}' "
                        f"but that module has been disallowed"
                    ))
                self._add_extern(extern_module)

            if len(self.errors):
                import ubelt as ub
                print('self.errors = {}'.format(ub.urepr(self.errors, nl=1)))

            for fname in self.zip_reader.get_all_records():
                self._add_file(fname)

            self.patched_builtins = builtins.__dict__.copy()
            self.patched_builtins["__import__"] = self.__import__
            # Allow packaged modules to reference their PackageImporter
            self.modules["torch_package_importer"] = self  # type: ignore[assignment]

            self._mangler = PackageMangler()

            # used for reduce deserializaiton
            self.storage_context: Any = None
            self.last_map_location = None

            # used for torch.serialization._load
            self.Unpickler = lambda *args, **kwargs: PackageUnpickler(self, *args, **kwargs)

        def _add_file(self, filename: str):
            """Assembles a Python module out of the given file. Will ignore files in the .data directory.

            Args:
                filename (str): the name of the file inside of the package archive to be added
            """
            *prefix, last = filename.split("/")
            if len(prefix) > 1 and prefix[0] == ".data":
                return
            package = self._get_or_create_package(prefix)
            errored = False
            if isinstance(package, _ExternNode):
                self.errors.append((
                    f"inconsistent module structure. package contains a module file {filename}"
                    f" that is a subpackage of a module marked external."
                ))
                errored = True
            if not errored:
                if last == "__init__.py":
                    package.source_file = filename
                elif last.endswith(".py"):
                    package_name = last[: -len(".py")]
                    package.children[package_name] = _ModuleNode(filename)

    from torch import package
    imp = CustomPackageImporter(package_path)
    # import json
    # header_infos = ['package_header']
    # header = json.loads(imp.zip_reader.read(header_infos['package_header.json']).decode('utf8').strip())

    # package_header = json.loads(imp.load_text(
    #     'package_header', 'package_header.json'))
    # arch_name = package_header['arch_name']
    # module_name = package_header['module_name']

    # # We can load the model through errors if our external state is actually ok
    # model = imp.load_pickle(module_name, arch_name)
    # state = model.state_dict()

    # # Can recreate an unmangled version of the model by constructing a new
    # # instance.
    # cls = type(model)
    # new_model = cls(**model.hparams)
    # new_model.load_state_dict(state)

    # new_model.save_package('foo.pt')
    # from geowatch.tasks.fusion.utils import load_model_from_package
    # recon_model = load_model_from_package('foo.pt')
    return imp


class Lambda(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return self.lambda_(x)


class DimensionDropout(nn.Module):
    def __init__(self, dim, n_keep):
        super().__init__()
        self.dim = dim
        self.n_keep = n_keep

    def forward(self, x):
        shape = x.shape
        dim_size = shape[self.dim]

        index = [slice(0, None)] * len(shape)
        index[self.dim] = torch.randperm(dim_size)[:self.n_keep]

        return x[index]


def ordinal_position_encoding(num_items, feat_size, method='sin', device='cpu'):
    """
    A positional encoding that represents ordinal

    Args:
        num_items (int): number of dimensions to be encoded (
            e.g. this is a spatial or temporal index)
        feat_size (int): this is the number of dimensions in the positional
             encoding generated for each dimension / item

    Example:
        >>> # Use 5 feature dimensions to encode 3 timesteps
        >>> from geowatch.tasks.fusion.utils import *  # NOQA
        >>> num_timesteps = num_items = 3
        >>> feat_size = 5
        >>> encoding = ordinal_position_encoding(num_items, feat_size)
    """
    assert method == 'sin'
    sf = 10000
    parts = []
    base = torch.arange(num_items, device=device)
    for idx in range(feat_size):
        exponent = (idx / feat_size)
        modulator = (1 / (sf ** exponent))
        theta = base * modulator
        if idx % 2 == 0:
            part = torch.sin(theta)
        else:
            part = torch.cos(theta)
        parts.append(part)
    encoding = torch.stack(parts, dim=1)
    return encoding


class SinePositionalEncoding(nn.Module):
    """
    Args:
        dest_dim (int): feature dimension to concat to
        dim_to_encode (int): dimension encoding is supposed to represent
        size (int): number of different encodings for the dim_to_encode

    Example:
        >>> from geowatch.tasks.fusion.utils import *  # NOQA
        >>> dest_dim = 3
        >>> dim_to_encode = 2
        >>> size = 8
        >>> self = SinePositionalEncoding(dest_dim, dim_to_encode, size=size)
        >>> x = torch.rand(3, 5, 7, 11, 13)
        >>> y = self(x)

    Ignore:
        >>> from geowatch.tasks.fusion.utils import *  # NOQA
        >>> self = SinePositionalEncoding(1, 0, size=8)
        >>> encoding = self._encoding_part(10)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> import pandas as pd
        >>> sns = kwplot.autosns()
        >>> df = pd.concat([pd.DataFrame({'x': np.arange(len(part)), 'y': part, 'part': [idx] * len(part)}) for idx, part in enumerate(encoding.T)]).reset_index()
        >>> fig = kwplot.figure(pnum=(1, 2, 1))
        >>> ax = fig.gca()
        >>> sns.lineplot(data=df, x='x', y='y', hue='part')
        >>> kwplot.imshow(kwarray.normalize(encoding.numpy()).T, pnum=(1, 2, 2), cmap='magma')
    """

    def __init__(self, dest_dim, dim_to_encode, size=4):
        super().__init__()
        self.dest_dim = dest_dim
        self.dim_to_encode = dim_to_encode
        self.size = size
        assert self.dest_dim != self.dim_to_encode

    def _encoding_part(self, num, device='cpu'):
        sf = 10000
        parts = []
        base = torch.arange(num, device=device)
        for idx in range(self.size):
            exponent = (idx / self.size)
            modulator = (1 / (sf ** exponent))
            theta = base * modulator
            if idx % 2 == 0:
                part = torch.sin(theta)
            else:
                part = torch.cos(theta)
            parts.append(part)
        encoding = torch.stack(parts, dim=1)
        return encoding

    @profile
    def forward(self, x):
        device = x.device
        expanded_shape = list(x.shape)
        expanded_shape[self.dest_dim] = -1

        expand_dims = [None] * len(expanded_shape)
        expand_dims[self.dim_to_encode] = slice(0, None)
        expand_dims[self.dest_dim] = slice(0, None)

        num = expanded_shape[self.dim_to_encode]

        encoding = self._encoding_part(num, device)
        encoding = encoding[expand_dims].expand(expanded_shape)

        x = torch.cat([x, encoding.type_as(x)], dim=self.dest_dim)
        return x


def model_json(model, max_depth=float('inf'), depth=0):
    """
    import torchvision
    model = torchvision.models.resnet50()
    info = model_json(model, max_depth=1)
    print(ub.urepr(info, sort=0, nl=-1))
    """
    info = {
        'type': model._get_name(),
    }
    params = model.extra_repr()
    if params:
        info['params'] = params

    if model._modules:
        if depth >= max_depth:
            info['children'] = ...
        else:
            children = {
                key: model_json(child, max_depth, depth=depth + 1)
                for key, child in model._modules.items()
            }
            info['children'] = children
    return info


@ub.memoize
def _memo_legend(label_to_color):
    import kwplot
    legend_img = kwplot.make_legend_img(label_to_color)
    return legend_img


def category_tree_ensure_color(classes):
    """
    Ensures that each category in a CategoryTree has a color

    TODO:
        - [ ] Add to CategoryTree
        - [ ] TODO: better function
        - [ ] Consolidate with ~/code/watch/geowatch/tasks/fusion/utils :: category_tree_ensure_color
        - [ ] Consolidate with ~/code/watch/geowatch/utils/kwcoco_extensions :: category_category_colors
        - [ ] Consolidate with ~/code/watch/geowatch/heuristics.py :: ensure_heuristic_category_tree_colors
        - [ ] Consolidate with ~/code/watch/geowatch/heuristics.py :: ensure_heuristic_coco_colors

    Example:
        >>> import kwcoco
        >>> classes = kwcoco.CategoryTree.demo()
        >>> assert not any('color' in data for data in classes.graph.nodes.values())
        >>> category_tree_ensure_color(classes)
        >>> assert all('color' in data for data in classes.graph.nodes.values())
    """
    backup_colors = iter(kwimage.Color.distinct(len(classes)))
    for node in classes.graph.nodes:
        color = classes.graph.nodes[node].get('color', None)
        if color is None:
            color = next(backup_colors)
            classes.graph.nodes[node]['color'] = kwimage.Color(color).as01()
