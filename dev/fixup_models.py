"""
This script will iterate through a hard-coded pattern of model files
and apply a hard coded search / replace fix to the text of the model code.

It then repackages, updates, and adds the fixed models to DVC.
"""
import watch
from kwcoco.util import util_archive
import ubelt as ub


def main():
    from xdev.patterns import Pattern
    import xdev as xd
    import re
    import zipfile
    import tempfile

    dvc_dpath = watch.find_smart_dvc_dpath()
    model_fpaths = list((dvc_dpath / 'models/fusion/eval3_candidates/packages/').glob('*Simplify*/*.pt'))

    lines_to_remove = [
        "print(f'sensor={sensor}')",
        "print(f'  * chan_code={chan_code}')"
    ]

    final_staging_dpath = ub.Path(tempfile.mkdtemp())

    to_replace = []
    for model_fpath in ub.ProgIter(model_fpaths, desc='modifying models'):
        archive = util_archive.Archive(model_fpath, backend=zipfile)
        names = list(archive)

        fixed_text = None
        fixed_name = None

        for name in names:
            if name.endswith('channelwise_transformer.py'):
                data = archive.file.read(name)
                orig_text = data.decode('utf8')
                text = orig_text
                for tofind in lines_to_remove:
                    pat = Pattern.coerce(re.escape(tofind), hint='regex')
                    text = pat.sub('...', text)
                if text != orig_text:
                    fixed_name = name
                    fixed_text = text
                    print(xd.difftext(text, orig_text, colored=True))

        if fixed_text is not None:
            # Reconstruct the modified archive

            # Extract all data to a temp directory
            tmpdpath = ub.Path(tempfile.mkdtemp())
            extracted_fpaths = archive.extractall(output_dpath=tmpdpath)

            # Modify the underlying data
            fixed_fpath = tmpdpath / fixed_name
            fixed_fpath.write_text(fixed_text)

            # Repackage as a new archive
            rel_model_fpath = model_fpath.relative_to(dvc_dpath)
            new_model_fpath = final_staging_dpath / rel_model_fpath
            new_model_fpath.parent.ensuredir()

            import os
            new_zipfile = zipfile.ZipFile(new_model_fpath, mode='w')
            with new_zipfile:
                for disk_fpath in extracted_fpaths:
                    disk_fpath = ub.Path(disk_fpath)
                    arcname = str(disk_fpath.relative_to(tmpdpath))
                    new_zipfile.write(os.fspath(disk_fpath), arcname=arcname)
            ub.delete(tmpdpath)
            to_replace.append((model_fpath, new_model_fpath))

    from watch.utils import simple_dvc
    dvc_model_fpaths = [a for a, b in to_replace]

    dvc = simple_dvc.SimpleDVC(dvc_dpath)
    dvc.unprotect(dvc_model_fpaths)

    import os
    import shutil
    for model_fpath, new_model_fpath in ub.ProgIter(to_replace, 'overwrite old models'):
        # ub.hash_data(new_model_fpath)
        # ub.hash_data(model_fpath)
        shutil.move(new_model_fpath, model_fpath)

    for model_fpath, new_model_fpath in ub.ProgIter(to_replace, 'checking'):
        assert model_fpath.exists()
        assert not new_model_fpath.exists()

    # Add modified files
    dvc.add(dvc_model_fpaths)
