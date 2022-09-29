
"""
Model packages were broken

Breaking model change happened on 2022-03-12


V149 - V154 were trained on the modified model

I manually fixed FUSION_EXPERIMENT_SC_DM_wv_p8_V133_epoch=120-step=123903.pt

"""
# Breaking model change
# FUSION_EXPERIMENT_ML_V149


def schedule_fixes():
    # Temporary job
    import watch
    dvc_dpath = watch.find_smart_dvc_dpath()
    package_dpath = dvc_dpath / 'models/fusion/eval3_candidates/packages'
    packages = list(package_dpath.glob('*/*.pt'))

    import datetime
    bad_time = datetime.datetime.fromisoformat('2022-03-12')

    items = []
    for fpath in packages:
        ver = int(fpath.parent.stem.split('_')[-1].lower().replace('v', ''))
        stat = fpath.stat()
        # ctime = datetime.datetime.fromtimestamp(stat.st_ctime)
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        items.append({
            'ver': ver,
            'mtime': mtime,
            'fpath': fpath,
        })

    sorted(items, key=lambda x: x['ver'])

    needs_fixes = []
    for item in items:
        if item['mtime'] > bad_time and item['ver'] < 149:
            needs_fixes.append(item)

    set([x['ver'] for x in needs_fixes])

    import tempfile
    import ubelt as ub
    temp_dir = tempfile.TemporaryDirectory()
    temp_dpath = ub.Path(temp_dir.name)

    from kwcoco.util import util_archive
    for item in needs_fixes:
        package_fpath = item['fpath']
        new_package_fpath = temp_dpath / package_fpath.name

        if 1:
            dpath = (temp_dpath / package_fpath.stem).ensuredir()
            print('dpath = {!r}'.format(dpath))

            archive = util_archive.Archive(package_fpath)
            extracted_files = archive.extractall(dpath)

            new_fpath = ub.Path(watch.tasks.fusion.methods.channelwise_transformer.__file__)

            patch_file = None
            for extract_fpath in extracted_files:
                if extract_fpath.endswith('watch/tasks/fusion/methods/channelwise_transformer.py'):
                    patch_file = ub.Path(extract_fpath)

            # Overwrite with the new file
            patch_file.write_text(new_fpath.read_text())

            # Repackage
            import zipfile
            new_archive = zipfile.ZipFile(new_package_fpath, 'w')
            for extract_fpath in extracted_files:
                extract_fpath = ub.Path(extract_fpath)
                arcname = extract_fpath.relative_to(dpath)
                new_archive.write(extract_fpath, arcname=arcname)
            new_archive.close()

        item['new_package_fpath'] = new_package_fpath

    # import sys, ubelt
    # from watch.tasks.fusion.repackage import SimpleDVC  # NOQA
    # SimpleDVC.add(staged_expt_dpaths)
    # SimpleDVC.push(storage_dpath, remote=dvc_remote)

    rel_paths = [str(item['fpath'].relative_to(dvc_dpath)) for item in needs_fixes]
    print('dvc unprotect ' + ' '.join(rel_paths))

    import shutil
    for item in needs_fixes:
        shutil.copy(
            item['new_package_fpath'],
            item['fpath']
        )

    print('dvc add ' + ' '.join(rel_paths))
