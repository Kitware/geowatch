

def __oneoff():
    # TODO: might make a general migrate function for the manager.
    from watch.mlops.expt_manager import ExperimentState
    import watch
    import ubelt as ub
    import os
    import shutil
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
    data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
    dataset_code = '*'
    dvc_remote = 'aws'
    self = ExperimentState(expt_dvc_dpath, dataset_code, dvc_remote, data_dvc_dpath)

    # self.summarize()
    # Migrate pixel metrics to the new home
    to_move = []
    for src_t, dst_t in self.legacy_versioned_templates:
        src_p = src_t.format(**self.patterns)
        dst_t.format(**self.patterns)
        from watch.utils import util_pattern
        src_paths = list(util_pattern.Pattern.coerce(src_p).paths())
        import parse
        for path in src_paths:
            row = {}
            template = src_t
            parser = parse.Parser(str(template))
            results = parser.parse(str(path))
            if results is None:
                raise AssertionError(path)
            if results is not None:
                row.update(results.named)
            else:
                raise Exception('bad attrs')
            new_path = dst_t.format(**row)
            path = ub.Path(path)
            new_path = ub.Path(new_path)
            to_move.append((path, new_path))

    # import subprocess
    for src, dst in ub.ProgIter(to_move):
        src = ub.Path(src)
        dst = ub.Path(dst)
        if src.exists() and not dst.exists():
            if not dst.parent.exists():
                dst.parent.ensuredir()
            info = ub.cmd(['git', 'mv', os.fspath(src), os.fspath(dst)], verbose=0, cwd=src.parent)
            if info['ret'] != 0:
                if 'source directory is empty' in info['err']:
                    shutil.move(src, dst)
                else:
                    raise AssertionError
        else:
            if not dst.exists():
                raise Exception
            if src.exists():
                info = ub.cmd(['git', 'mv', os.fspath(src), os.fspath(dst)], verbose=0, cwd=src.parent)
                if info['ret'] != 0:
                    if 'source directory is empty' in info['err']:
                        shutil.move(src, dst)
                    else:
                        raise AssertionError('what')

    for src, dst in ub.ProgIter(to_move, desc='check'):
        src = ub.Path(src)
        dst = ub.Path(dst)
        assert dst.exists()
        assert not src.exists()

    # for src, dst in ub.ProgIter(to_move, desc='check', verbose=3):
    #     src = ub.Path(src)
    #     dst = ub.Path(dst)
    #     if not src.exists() and dst.exists():
    #         shutil.move(dst, src)
    #     else:
    #         print('bad')

    import glob
    curve_paths = glob.glob(str(ub.Path(self.path_patterns['eval_pxl']).parent))
    for dpath in curve_paths:
        dpath = ub.Path(dpath)
        whoops = (dpath / 'curves')
        if whoops.exists():
            if len(whoops.ls()):
                ub.cmd('mv * ..', cwd=whoops, shell=True, verbose=1, check=True)
            assert len(whoops.ls()) == 0
            whoops.rmdir()

    to_delete = []
    for r, ds, fs in (self.expt_dvc_dpath / 'models').walk():
        if 'tmp' in ds:
            ds.remove('tmp')
            assert 'eval' in r.name
            to_delete.append(r / 'tmp')

    for p in ub.ProgIter(to_delete, desc='removing temp dirs'):
        p.delete()

    import glob
    trk_paths = glob.glob(str(ub.Path(self.path_patterns['eval_trk']).parent))
    for p in trk_paths:
        p = ub.Path(p)
        if (p / 'tmp').exists():
            raise Exception
        print(p.parent.parent.ls())
