"""
Script for porting relevant parts of submodules to a static path in this repo.

Ideally users of this repo do not need to ever touch submodules, but as we
develop I do want to ensure we tie TPL code to the original source as best as
possible (and contribute back!).
"""


def main():
    import geowatch_tpl
    import ubelt as ub
    import rich

    tpl_dpath = ub.Path(geowatch_tpl.__file__).parent
    dev_submod_dpath = tpl_dpath / 'submodules'
    static_submod_dpath = tpl_dpath / 'submodules_static'
    rich.print('dev_submod_dpath = {}'.format(ub.urepr(dev_submod_dpath, nl=1)))
    rich.print('static_submod_dpath = {}'.format(ub.urepr(static_submod_dpath, nl=1)))

    SUBMODULE_INFOS = list(geowatch_tpl.STATIC_SUBMODULES.items())
    rich.print('SUBMODULE_INFOS = {}'.format(ub.urepr(SUBMODULE_INFOS, nl=1)))

    rich.print('[green]--- Copy Submodules ---\n')

    for key, info in ub.ProgIter(SUBMODULE_INFOS, desc='copy submodules', verbose=3):
        rel_dpath = info['rel_dpath']
        src_dpath = dev_submod_dpath / rel_dpath
        dst_dpath = static_submod_dpath / rel_dpath

        if src_dpath.exists():
            rich.print(f'Copy: {src_dpath} -> {dst_dpath}')
            dst_dpath.delete()
            dst_dpath.parent.ensuredir()
            if 'parts' in info:
                for p in info['parts']:
                    p_src = dev_submod_dpath / p
                    print(f'p_src={p_src}')
                    p_dst = static_submod_dpath / p
                    assert p_src.exists()
                    p_dst.parent.ensuredir()
                    p_dst.delete()
                    p_src.copy(p_dst)
                    ...
            else:
                src_dpath.copy(dst_dpath)

            if 'ignore' in info:
                # Remove parts that were ignored. (dont have a good way to
                # exclude them from copy itself)
                for p in info['ignore']:
                    ignore_path = static_submod_dpath / p
                    print(f'ignore_path={ignore_path}')
                    ignore_path.delete()

        else:
            print(f'No dev: {src_dpath}')
        rich.print('[yellow]---\n')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch_tpl/snapshot_submodules.py
    """
    main()
