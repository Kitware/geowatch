#!/usr/bin/env python3
"""
Ignore:
    python ~/code/watch/dev/poc/delocalize_command.py -- python -m geowatch.tasks.fusion.predict --package_fpath=/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V33/Drop6_BAS_scratch_landcover_10GSD_split2_V33_epoch604_step38115.pt --test_dataset=/home/joncrall/remote/namek/data/dvc-repos/smart_data_dvc/ValiRegionSmall/combo_small_NZ_R001_swnykmah_I2.kwcoco.zip --pred_dataset=/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/_namek_split2_eval_small/pred/flat/bas_pxl/bas_pxl_id_b742b70c/pred.kwcoco.zip --chip_overlap=0.3 --chip_dims=auto --time_span=auto --time_sampling=auto --drop_unused_frames=True --num_workers=2 --devices=0, --batch_size=1 --with_saliency=True --with_class=False


    python ~/code/watch/dev/poc/delocalize_command.py -- python -m geowatch.tasks.fusion.predict --package_fpath=/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2_epoch16_step1700.pt --test_dataset=/home/joncrall/remote/namek/data/dvc-repos/smart_data_dvc/ValiRegionSmall/combo_small_NZ_R001_swnykmah_I2.kwcoco.zip --pred_dataset=/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/_namek_split2_eval_small/pred/flat/bas_pxl/bas_pxl_id_eaba5207/pred.kwcoco.zip --chip_overlap=0.3 --chip_dims=auto --time_span=auto --time_sampling=auto --drop_unused_frames=True --num_workers=2 --devices=0, --batch_size=1 --with_saliency=True --with_class=False --with_change=False


"""
import os
import scriptconfig as scfg
import ubelt as ub


class DelocalizeCommandConfig(scfg.DataConfig):
    """
    Attempts to delocalize a command by replacing absolute paths
    with ones relative to known special environment variables.

    TODO:
        - [ ] General path registry
        - [ ] Better heuristics
    """
    command = scfg.Value('command', type=str, position=1, nargs='*')


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict(
        >>> )
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = DelocalizeCommandConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    print('config = ' + ub.urepr(dict(config), nl=1))
    command = config.command

    # Split --key=value arguments
    import parse
    keyval_parser = parse.Parser('--{key}={value}')
    normalized = []
    for part in command:
        parsed = keyval_parser.parse(part)
        if parsed:
            normalized.append('--{key}'.format(**parsed.named))
            normalized.append('{value}'.format(**parsed.named))
        else:
            normalized.append(part)

    # Check
    import watch
    data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    expt_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')

    replaceable_paths = {
        'DVC_DATA_DPATH': data_dpath,
        'DVC_EXPT_DPATH': expt_dpath,
    }

    delocalized_parts = []
    for part in normalized:
        if maybe_path(part):
            path = part
            key, path, tail = delocalize_path(path, replaceable_paths)
            if key is not None:
                path = '${' + key + '}/' + os.fspath(tail)
                delocalized_parts.append(os.fspath(path))
            else:
                delocalized_parts.append(os.fspath(path))
        else:
            delocalized_parts.append(part)

    import shlex
    print(ub.codeblock(
        f'''
        DVC_DATA_DPATH={data_dpath}
        DVC_EXPT_DPATH={expt_dpath}
        '''))
    print(ub.codeblock(
        '''
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
        DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
        '''))
    # print(shlex.join(delocalized_parts))
    print(' '.join(delocalized_parts))


def maybe_path(p):
    # TODO: better heuristic
    if '/' in str(p):
        return True


def delocalize_path(path, replaceable_paths):
    orig = ub.Path(path)
    for i in range(2, len(orig.parts)):
        for key, base in replaceable_paths.items():
            tail = ub.Path(*orig.parts[i:])
            cand = base / tail
            if cand.exists():
                found = cand
                return key, found, tail

    known_relative_paths = {
        'DVC_DATA_DPATH': '/home/joncrall/remote/namek/data/dvc-repos/smart_data_dvc',
        'DVC_EXPT_DPATH': '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc',
    }
    for key, cand_base in known_relative_paths.items():
        if str(orig).startswith(cand_base):
            tail = orig.relative_to(cand_base)
            found = base / tail
            return key, found, tail
    return None, path, None

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/poc/delocalize_command.py
        python -m delocalize_command
    """
    main()
