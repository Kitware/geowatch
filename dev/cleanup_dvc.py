def _cleanup_extra_versions_in_dvc():
    import ubelt as ub
    import re
    dpath = ub.Path('$HOME/data/dvc-repos/smart_watch_dvc/models/fusion/eval3_candidates/packages').expand()

    package_fpaths = list(dpath.glob('*/*.pt.dvc'))
    # Discard the -v2, -v3, etc... paths if a different one exists
    def remove_v_suffix(x):
        return re.sub(r'-v[0-9]+$', '', x.stem, flags=re.MULTILINE)
    unique_package_fpaths = list(ub.unique(
        sorted(package_fpaths), key=remove_v_suffix))
    dup_packages = set(package_fpaths) - set(unique_package_fpaths)  # NOQA

    dvc_package_fpaths = list(dpath.glob('*/*.pt.dvc'))
    def remove_v_dvc_suffix(x):
        return re.sub(r'-v[0-9]+$', '', x.stem.replace('.pt', ''), flags=re.MULTILINE)
    unique_dvc_package_fpaths = list(ub.unique(
        sorted(dvc_package_fpaths), key=remove_v_dvc_suffix))
    dup_dvc_packages = set(dvc_package_fpaths) - set(unique_dvc_package_fpaths)

    tracked_to_remove = []
    for dup_dvc_pkg in dup_dvc_packages:
        pkg_path = dup_dvc_pkg.augment(ext='')
        if pkg_path.exists():
            tracked_to_remove.append(pkg_path)

    for p in tracked_to_remove:
        p.delete()

    for p in dup_dvc_packages:
        p.delete()
