# DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)


def codetemplate(text, format=False):
    import string
    import ubelt as ub
    import operator as op
    from functools import reduce
    code_text = ub.codeblock(text)
    template = string.Template(code_text)
    existing_vars = {reduce(op.add, t, '') for t in template.pattern.findall(text)}
    if format is False:
        return code_text
    elif format is True:
        import xdev
        format = xdev.get_stack_frame(1).f_locals
    fmtdict = ub.udict(format) & existing_vars
    return template.safe_substitute(**fmtdict)


def main():
    import watch

    dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    all_regions = [p.name.split('.')[0] for p in (dvc_data_dpath / 'annotations/drop6/region_models').ls()]

    # time_duration = '1year'
    # time_duration = '3months'

    all_regions = [
        'KR_R001',
        'KR_R002',
        'NZ_R001',
        'CH_R001',
        'BR_R001',
        'BR_R002',
        'BH_R001',
    ]
    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', size=4,
                                   name='combine-time-queue')
    for region in all_regions:

        fmtdict = dict(
            DVC_DATA_DPATH=dvc_data_dpath,
            REGION=region,
            TIME_DURATION='1year',
            SUFFIX='MeanYear',
            # TIME_DURATION='3months',
            # SUFFIX='Mean3Month10GSD',
            RESOLUTION='10GSD',
        )

        code = codetemplate(
            r'''
            python -m watch.cli.coco_temporally_combine_channels \
                --kwcoco_fpath="$DVC_DATA_DPATH/Drop6/imgonly-${REGION}.kwcoco.json" \
                --output_kwcoco_fpath="$DVC_DATA_DPATH/Drop6_${SUFFIX}/imgonly-${REGION}.kwcoco.json" \
                --channels="red|green|blue|nir|swir16|swir22" \
                --resolution="${RESOLUTION}" \
                --temporal_window_duration=$TIME_DURATION \
                --merge_method=mean \
                --workers=4
            ''', fmtdict)
        combine_job = queue.submit(code, name=f'combine-time-{region}')
        # combine_job = None

        code = codetemplate(
            r'''
            python -m watch reproject_annotations \
                --src $DVC_DATA_DPATH/Drop6_${SUFFIX}/imgonly-${REGION}.kwcoco.json \
                --dst $DVC_DATA_DPATH/Drop6_${SUFFIX}/imganns-${REGION}.kwcoco.zip \
                --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/*.geojson"
            ''', fmtdict)
        queue.submit(code, depends=[combine_job], name=f'reproject-ann-{region}')

    queue.print_commands()
    queue.print_graph()
    queue.run()
