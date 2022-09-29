
import pathlib
import ubelt as ub
from os.path import relpath
from kwcoco.util import util_archive

dpath = pathlib.Path('/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/activity')
package_fpaths = list(dpath.glob('*.pt'))

for fpath in package_fpaths:
    ub.cmd('dvc unprotect ' + str(fpath), cwd=dpath, verbose=3)
    # import arlib
    # archive = arlib.open(fpath)
    # zfile = archive._file

    # Should fail
    try:
        import torch
        imp = torch.package.PackageImporter(fpath)
    except Exception:
        pass
    else:
        raise AssertionError('should have failed')

    package_header = imp.load_pickle(
        'kitware_package_header', 'kitware_package_header.pkl')
    arch_name = package_header['arch_name']
    module_name = package_header['module_name']

    import tempdir
    import tempfile
    tmp = tempdir.TempDir()
    tmp.name

    self = util_archive.Archive(fpath, 'r')
    abs_members = self.extractall(tmp.name)
    rel_members = [relpath(p, tmp.name) for p in abs_members]

    zfile = self.file
    bad_members = [name for name in rel_members if '/' not in name]

    good_member = pathlib.Path(ub.peek(set(rel_members) - set(bad_members)))
    topdir = pathlib.Path(good_member.parts[0])

    # "Move" the file in the archive
    tmp_dpath = pathlib.Path(tmp.name)
    import xdev as xd
    xd.tree(tmp_dpath, max_files=10)

    import shutil
    for rel_src in bad_members:
        src = tmp_dpath / rel_src
        dst = tmp_dpath / topdir / 'kitware_package_header' / rel_src
        dst.parent.mkdir(exist_ok=True, parents=True)
        shutil.move(src, dst)

    tmpf = tempfile.NamedTemporaryFile(suffix='.zip')
    new_fpath = tmpf.name
    ub.delete(new_fpath)
    new_ar = util_archive.Archive(new_fpath, 'w')

    abs_members = list(tmp_dpath.glob('**/*'))
    for member_fpath in abs_members:
        arcname = member_fpath.relative_to(tmp_dpath)
        new_ar.add(member_fpath, arcname)
    new_ar.close()

    shutil.move(new_fpath, fpath)

    # Test new one works
    imp = torch.package.PackageImporter(fpath)
    yaml_text = imp.load_text('kitware_package_header', 'fit_config.yaml')
    package_header = imp.load_pickle(
        'kitware_package_header', 'kitware_package_header.pkl')
    arch_name = package_header['arch_name']
    module_name = package_header['module_name']


# Fix packages
for fpath in package_fpaths:

    imp = torch.package.PackageImporter(fpath)
    package_header = imp.load_pickle(
        'kitware_package_header', 'kitware_package_header.pkl')

    import json
    json.dumps(package_header)
