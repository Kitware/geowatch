
def main():
    from watch.tasks.fusion.utils import load_model_from_package
    import watch
    from watch.monkey import monkey_torch
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')

    package_path = expt_dvc_dpath / 'models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt'
    model_works = load_model_from_package(package_path)
    monkey_torch.fix_gelu_issue(model_works)
    print(f'model_works={model_works}')

    fpath = expt_dvc_dpath / 'models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch44_step46014.pt'
    model = load_model_from_package(fpath)

    from torch import package
    imp = package.PackageImporter(fpath)


def debug_torch_package(fpath):
    import zipfile
    import ubelt as ub
    zfile = zipfile.ZipFile(fpath)

    nesting = ub.AutoDict()
    walker = ub.IndexableWalker(nesting)
    infos = zfile.infolist()
    for info in infos:
        parts = info.filename.split('/')
        walker[parts] = info
    nesting = nesting.to_dict()

    top_level_paths = list(nesting.keys())
    print('top_level_paths = {}'.format(ub.urepr(top_level_paths, nl=1)))
    if len(top_level_paths) != 1:
        print('Should only have one top level path')

    top_level_pat = top_level_paths[0]
    top_level = nesting[top_level_pat]

    print('nesting = {}'.format(ub.urepr(nesting, nl=True)))

    data_infos = top_level['.data']
    extern_module_info = data_infos['extern_modules']
    extern_modules = sorted([n.strip() for n in zfile.read(extern_module_info).decode('utf8').split('\n') if n.strip()])
    py_version = zfile.read(data_infos['python_version']).decode('utf8')
    pkg_version = zfile.read(data_infos['version']).decode('utf8').strip()

    print('extern_modules = {}'.format(ub.urepr(extern_modules, nl=1)))
    print('py_version = {}'.format(ub.urepr(py_version, nl=1)))
    print('pkg_version = {}'.format(ub.urepr(pkg_version, nl=1)))

    import json
    header_infos = top_level['package_header']
    header = json.loads(zfile.read(header_infos['package_header.json']).decode('utf8').strip())
    print('header = {}'.format(ub.urepr(header, nl=1)))
    # pkl_model_info = top_level[header['module_name']][header['arch_name']]
    # pkl_data = zfile.read(pkl_model_info)

    # class PackageUnpickler(pickle._Unpickler):  # type: ignore[name-defined]
    #     """Package-aware unpickler.

    #     This behaves the same as a normal unpickler, except it uses `importer` to
    #     find any global names that it encounters while unpickling.
    #     """

    #     def __init__(self, importer: Importer, *args, **kwargs):
    #         super().__init__(*args, **kwargs)
    #         self._importer = importer

    #     def find_class(self, module, name):
    #         # Subclasses may override this.
    #         if self.proto < 3 and self.fix_imports:  # type: ignore[attr-defined]
    #             if (module, name) in _compat_pickle.NAME_MAPPING:
    #                 module, name = _compat_pickle.NAME_MAPPING[(module, name)]
    #             elif module in _compat_pickle.IMPORT_MAPPING:
    #                 module = _compat_pickle.IMPORT_MAPPING[module]
    #         mod = self._importer.import_module(module)
    #         return getattr(mod, name)
    # import pickle
    # pickle.loads(pkl_data)


def tryfix_torch_package(fpath):
    import zipfile
    import ubelt as ub

    new_fpath = 'tryfix.pt'

    blocklist = {
        'watch/tasks/fusion/methods/channelwise_transformer.py'
        'watch/tasks/fusion/utils.py'
    }

    src_zfile = zipfile.ZipFile(fpath, 'r')
    dst_zfile = zipfile.ZipFile(new_fpath, 'w')

    top_level_cands = set()
    for info in src_zfile.infolist():
        fname = ub.Path(info.filename)
        top_level_cands.add(fname.parts[0])

    assert len(top_level_cands) == 1
    top_level_cand = ub.Path(ub.peek(top_level_cands))
    real_blocklist = {
        top_level_cand / b for b in blocklist
    }
    with dst_zfile:
        for info in src_zfile.infolist():
            fname = ub.Path(info.filename)
            if fname not in real_blocklist:
                raw_data = src_zfile.read(info)
                dst_zfile.writestr(info.filename, raw_data)

    from torch import package
    imp = package.PackageImporter(new_fpath)


def broken_package_details(fpath):
    """
    Modifies PackageImporter to get better errror messages.
    Could be a PR to torch.
    """
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
    imp = CustomPackageImporter(fpath)

    import json
    header_infos = ['package_header']
    header = json.loads(imp.zip_reader.read(header_infos['package_header.json']).decode('utf8').strip())

    package_header = json.loads(imp.load_text(
        'package_header', 'package_header.json'))
    arch_name = package_header['arch_name']
    module_name = package_header['module_name']

    # We can load the model through errors if our external state is actually ok
    model = imp.load_pickle(module_name, arch_name)
    state = model.state_dict()

    # Can recreate an unmangled version of the model by constructing a new
    # instance.
    cls = type(model)
    new_model = cls(**model.hparams)
    new_model.load_state_dict(state)

    new_model.save_package('foo.pt')
    from watch.tasks.fusion.utils import load_model_from_package
    recon_model = load_model_from_package('foo.pt')
