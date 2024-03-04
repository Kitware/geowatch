#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class FixupPredictKwcocoMetadata(scfg.DataConfig):
    """
    Update pre-0.15.1 kwcoco predictions to properly store train-time params.

    The kwcoco info section of kwcoco files produced by geowatch.fusion.predict
    only contained the "data" section of the fit configuration. This script is
    able to fix one or more of those old predicted files as long as the path to
    the model is available. Warnings that direct users to this help document
    will typically give example usage that fixes one file, but multiple files
    can be fixed at once by specifying a glob pattern. The following example
    illustrates this.

    CommandLine:
        # Say you have an old mlops directory of results
        DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
        MLOPS_DPATH=$DVC_EXPT_DPATH/_preeval20_bas_grid

        # Construct a glob pattern that matches the kwcoco files that need to
        # be fixed, and pass it to this script.
        python -m geowatch.cli.experimental.fixup_predict_kwcoco_metadata \\
            --coco_fpaths "$MLOPS_DPATH/pred/flat/bas_pxl/*/pred.kwcoco.zip"

    Note:
        This modifies input files INPLACE!
    """
    coco_fpaths = scfg.Value(None, help='Path to one or more predicted kwcoco files to fix')
    workers = scfg.Value(8)

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from fixup_bad_fit_config import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = FixupBadFitConfigCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))

        from kwutil import util_path
        from kwutil import util_progress

        kwcoco_fpaths = util_path.coerce_patterned_paths(config.coco_fpaths)
        pman = util_progress.ProgressManager()
        jobs = ub.JobPool(mode='process', max_workers=config.workers)
        with pman, jobs:
            for node_dpath in pman.progiter(kwcoco_fpaths, desc='submit kwcoco fixup jobs'):
                jobs.submit(fixup_pxl_pred_node_dpath, node_dpath)

            stats = ub.ddict(int)
            for job in pman.progiter(jobs.as_completed(), desc='collect fixup jobs'):
                status = job.result()
                stats[status['message']] += 1
                pman.update_info(f'stats = {ub.urepr(stats, nl=1)}')


def fixup_pxl_pred_node_dpath(coco_fpath):
    import kwcoco
    from kwutil.util_yaml import Yaml
    import zipfile
    status = {}
    messages = []
    dset = kwcoco.CocoDataset(coco_fpath)
    candidates = []
    for item in dset.dataset['info']:
        if item['type'] == 'process':
            if item['properties']['name'] == 'geowatch.tasks.fusion.predict':
                candidates.append(item)
    assert len(candidates) == 1
    pred_item = candidates[0]
    extra = pred_item['properties']['extra']
    if 'fit_config' in extra:
        if set(extra['fit_config']).issuperset({'data', 'trainer'}):
            messages.append('Already had updated fit config')
        else:
            status['message'] = 'Updated metadata'
            package_fpath = pred_item['properties']['config']['package_fpath']
            zfile = zipfile.ZipFile(package_fpath)
            fit_config = None
            with zfile:
                for name in zfile.namelist():
                    if name.endswith('package_header/config.yaml'):
                        config_text = zfile.read(name).decode('utf8')
                        fit_config = Yaml.loads(config_text, backend='pyyaml')
                        break
            assert fit_config is not None
            extra['fit_config'] = fit_config
            dset.dump()
            messages.append('Updated metadata')
    status['message'] = '\n'.join(messages)
    return status

__cli__ = FixupPredictKwcocoMetadata
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/dev/oneoffs/fixup_predict_kwcoco_metadata.py
    """
    main()
