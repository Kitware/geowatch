"""
Script for porting relevant parts of submodules to a static path in this repo.

Ideally users of this repo do not need to ever touch submodules, but as we
develop I do want to ensure we tie TPL code to the original source as best as
possible (and contribute back!).
"""


def main():
    import geowatch_tpl
    import ubelt as ub

    tpl_dpath = ub.Path(geowatch_tpl.__file__).parent
    dev_submod_dpath = tpl_dpath / 'submodules'
    static_submod_dpath = tpl_dpath / 'submodules_static'
    print('dev_submod_dpath = {}'.format(ub.urepr(dev_submod_dpath, nl=1)))
    print('static_submod_dpath = {}'.format(ub.urepr(static_submod_dpath, nl=1)))

    for key, info in ub.ProgIter(list(geowatch_tpl.STATIC_SUBMODULES.items()), desc='copy submodules', verbose=3):
        rel_dpath = info['rel_dpath']
        src_dpath = dev_submod_dpath / rel_dpath
        dst_dpath = static_submod_dpath / rel_dpath

        if src_dpath.exists():
            print(f'{src_dpath} -> {dst_dpath}')
            dst_dpath.delete()
            dst_dpath.parent.ensuredir()
            if 'parts' in info:
                for p in info['parts']:
                    p_src = dev_submod_dpath / p
                    print(f'p_src={p_src}')
                    p_dst = static_submod_dpath / p
                    assert p_src.exists()
                    p_dst.parent.ensuredir()
                    p_src.copy(p_dst)
                    ...
            else:
                src_dpath.copy(dst_dpath)
        else:
            print(f'No dev: {src_dpath}')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch_tpl/snapshot_submodules.py
    """
    main()
