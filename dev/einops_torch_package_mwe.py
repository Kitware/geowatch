"""
# TODO: REPRODUCE AND REPORT https://github.com/arogozhnikov/einops/issues

Steps to reproduce the behavior:

[BUG] Issue with Einops Rearrange when using torch.package


**Describe the bug**

When using torch.package (torch 1.9)

**Reproduction steps**

**Expected behavior**
What you expected to happen.

**Your platform**
Version of einops, python and DL package that you used



CommandLine:
    mkdir -p einops_issue_155a
    touch einops_issue_155a/__init__.py
    cp einops_torch_package_mwe.py einops_issue_155a/mwe.py

    python einops_torch_package_mwe.py --action=main
    7z l foo.pt
    python einops_torch_package_mwe.py --action=load_package
"""

from einops.layers.torch import Rearrange
import torch
import json
import pytorch_lightning as pl


class DummyModule(pl.LightningModule):
    def __init__(self, param1=3, param2=5, mode='auto'):
        super().__init__()
        moddict = torch.nn.ModuleDict()
        self.save_hyperparameters()
        self.param1 = param1
        self.param2 = param2
        self.mode = mode
        ws = 8
        hs = 8
        moddict['a'] = Rearrange("b t c (h hs) (w ws) -> b t c h w (ws hs)", hs=hs, ws=ws)
        moddict['b'] = Rearrange("b t c (h hs) (w ws) -> b t c h w (ws hs)", hs=hs, ws=ws)
        self.layer = Rearrange("b t c h w -> b (c t h w)")
        self.moddict = moddict

    def forward(self, x):
        print('x.shape = {!r}'.format(x.shape))
        self.moddict['a'](x)
        self.moddict['b'](x)
        y = self.layer(x)
        print('y.shape = {!r}'.format(y.shape))
        print('y = {!r}'.format(y))


def save_package():
    import torch.package
    from einops_issue_155a import mwe
    model = orig = mwe.DummyModule()
    print('orig = {!r}'.format(orig))
    x = torch.rand(1, 2, 3, 32, 32)
    orig.forward(x)

    package_path = 'foo.pt'

    with torch.package.PackageExporter(package_path) as exp:
        exp.extern("**", exclude=["einops_issue_155a.**"])
        exp.intern("einops_issue_155a.**", allow_empty=False)
        arch_name = "model.pkl"
        module_name = 'watch_tasks_fusion'
        # new encoding
        package_header = {
            'version': '0.1.0',
            'arch_name': arch_name,
            'module_name': module_name,
        }

        exp.save_text(
            'package_header', 'package_header.json',
            json.dumps(package_header)
        )
        exp.save_pickle(module_name, arch_name, model)


def load_package():
    import torch.package
    package_path = 'foo.pt'
    imp = torch.package.PackageImporter(package_path)
    package_header = json.loads(imp.load_text('package_header', 'package_header.json'))
    arch_name = package_header['arch_name']
    module_name = package_header['module_name']
    recon = imp.load_pickle(module_name, arch_name)
    x = torch.rand(1, 2, 3, 32, 32)
    recon.forward(x)


def main():
    save_package()
    load_package()


if __name__ == '__main__':
    import ubelt as ub
    action = ub.argval('--action', default='main')
    func = vars()[action]
    func()
