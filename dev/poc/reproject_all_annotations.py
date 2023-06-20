#!/usr/bin/env python3
"""
SeeAlso:
    * ~/code/watch/dev/oneoffs/drop6_fixups.py
    * ~/code/watch/dev/poc/prepare_time_combined_dataset.py
"""
import scriptconfig as scfg
import ubelt as ub
from cmd_queue.cli_boilerplate import CMDQueueConfig


class ReprojectAllConfig(CMDQueueConfig):
    """
    Helper script to reproject annotations onto the coco files in a kwcoco
    bundle.
    """
    bundle_dpath = scfg.Value(None, help='the kwcoco bundle with region annotations to fixup')
    true_site_dpath = scfg.Value(None, help='path to the truth geojson site directory')
    true_region_dpath = scfg.Value(None, help='path to the truth geojson region directory')


@ub.memoize
def get_region_pattern():
    import re
    if 0:
        import xdev
        b = xdev.regex_builder.RegexBuilder.coerce('python')
        patparts = [
            b.named_field(r'.*[-_]', 'before'),
            b.named_field(r'[A-Z][A-Z]_[RC]\d\d\d', 'region_id'),
            b.named_field(r'[-_].*', 'after') + '?',
            b.named_field(r'\.kwcoco', 'kwcoco_ext'),
            b.named_field(r'\.[a-z]+$', 'ext'),
        ]
        pat = ''.join(patparts)
        pat = re.compile(pat)
        print('patparts = ' + ub.urepr(patparts))
    patparts = [
        '(?P<before>.*[-_])',
        '(?P<region_id>[A-Z][A-Z]_[RC]\\d\\d\\d)',
        '(?P<after>[-_].*)?',
        '(?P<kwcoco_ext>\\.kwcoco)',
        '(?P<ext>\\.[a-z]+$)',
    ]
    pat = re.compile(''.join(patparts))
    return pat


def heuristic_region_for_coco_fpath(fpath):
    pat = get_region_pattern()
    match = pat.match(fpath.name)
    parts = match.groupdict()
    region_id = parts['region_id']
    return region_id


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict()
        >>> main(cmdline=cmdline, **kwargs)
    """
    import rich
    config = ReprojectAllConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    rich.print('config = ' + ub.urepr(config, nl=1))
    config.bundle_dpath

    multi_region_fpaths = []
    single_region_fpaths = []
    unused_fpaths = []

    kwcoco_fpaths = ub.Path(config.bundle_dpath).ls('*.kwcoco.*')
    for fpath in kwcoco_fpaths:
        if '_train_' in fpath.name or '_vali_' in fpath.name or 'data.kwcoco' in fpath.name:
            # All regions
            multi_region_fpaths.append(fpath)
        elif 'imganns' in fpath.name:
            # heuristic for specific region
            single_region_fpaths.append(fpath)
        else:
            unused_fpaths.append(fpath)

    queue = config.create_queue()
    from kwutil import util_path
    for fpath in single_region_fpaths:
        region_id = heuristic_region_for_coco_fpath(fpath)
        region_fpath = (ub.Path(config.true_region_dpath) / (region_id + '.geojson'))
        site_globstr = (ub.Path(config.true_site_dpath) / (region_id + '_*.geojson'))
        assert region_fpath.exists()
        if '_R' in region_id:
            assert len(util_path.coerce_patterned_paths(site_globstr)) > 0

        code = ub.codeblock(
            fr'''
            python -m watch add_fields \
                --src {fpath} \
                --dst {fpath}
            ''')
        field_job = queue.submit(code, depends=[], name=f'add-fields-{fpath.name}')
        code = ub.codeblock(
            fr'''
            python -m watch reproject \
                --src {fpath} \
                --dst {fpath} \
                --region_models='{region_fpath}' \
                --site_models='{site_globstr}'
            ''')
        queue.submit(code, depends=[field_job], name=f'reproject-ann-{fpath.name}')

    for fpath in multi_region_fpaths:
        code = ub.codeblock(
            fr'''
            python -m watch add_fields \
                --src {fpath} \
                --dst {fpath}
            ''')
        field_job = queue.submit(code, depends=[], name=f'add-fields-{fpath.name}')
        code = ub.codeblock(
            fr'''
            python -m watch reproject \
                --src {fpath} \
                --dst {fpath} \
                --region_models='{config.true_region_dpath}' \
                --site_models='{config.true_site_dpath}'
            ''')
        queue.submit(code, depends=[field_job], name=f'reproject-ann-{fpath.name}')

    config.run_queue(queue)


if __name__ == '__main__':
    """

    CommandLine:

        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
        DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

        python ~/code/watch/dev/poc/reproject_all_annotations.py \
            --bundle_dpath $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2 \
            --true_site_dpath=$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models \
            --true_region_dpath=$DVC_DATA_DPATH/annotations/drop6_hard_v1/region_models \
            --backend=tmux \
            --tmux_workers=4 \
            --print_commands=1 --run=1
    """
    main()
