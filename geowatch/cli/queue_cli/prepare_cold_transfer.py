"""
Handle the batch transfer of COLD features to time averaged data

Ignore:

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
    python ~/code/watch/geowatch/cli/queue_cli/prepare_cold_transfer.py \
        --src_kwcocos "$DVC_DATA_DPATH/Aligned-Drop7/*/*cold.kwcoco.zip" \
        --dst_kwcocos "$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/*_I2LS.kwcoco.zip" \
        --run=1


    geowatch stats combo_imganns-KR_R001_I2LSC.kwcoco.zip

    geowatch visualize \
        combo_imganns-KR_R001_I2LSC.kwcoco.zip \
        --smart=True \
        --channels="(red|green|blue,pan,red_COLD_a1|green_COLD_a1|blue_COLD_a1,red_COLD_cv|green_COLD_cv|blue_COLD_cv,red_COLD_rmse|green_COLD_rmse|blue_COLD_rmse,sam.0:3,landcover_hidden.0:3,invariants.0:3)"

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
    python -m geowatch.cli.queue_cli.prepare_splits \
        --base_fpath=$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns*_I2LSC*.kwcoco.zip \
        --suffix=I2LSC \
        --backend=tmux --tmux_workers=6 \
        --run=1
"""
#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub
from cmd_queue.cli_boilerplate import CMDQueueConfig


class PrepareColdTransferConfig(CMDQueueConfig):
    """
    Run geowatch.tasks.cold.transfer_features on multiple regions in a cmd-queue
    """
    src_kwcocos = scfg.Value(None, help='input pattern for cold kwcoco files')
    dst_kwcocos = scfg.Value(None, help='pattern for cold files to transfer onto. Note this is *not* the output')
    new_suffix = scfg.Value('C', help='the suffix feature char code to append at the end of the coco name. Defaults to C for COLD')

    skip_existing = scfg.Value(True)
    cache = scfg.Value(True)


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict()
        >>> kwargs = {
        >>>     'src_kwcocos': '/home/joncrall/remote/Ooo/data/dvc-repos/smart_data_dvc/Aligned-Drop7/*/*cold.kwcoco.zip',
        >>>     'dst_kwcocos': '/home/joncrall/remote/Ooo/data/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD/*_I2L.kwcoco.zip',
        >>> }
        >>> main(cmdline=cmdline, **kwargs)
    """
    import rich
    from kwutil import util_path
    config = PrepareColdTransferConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    rich.print('config = ' + ub.urepr(config, nl=1))

    # Heuristic to extract region ids from file paths. This is not robust.
    import re
    region_id_pat = r'(?P<region_id>[A-Z]{2}_[CRT]\d{3})'
    prefix_pat = r'(?P<prefix>.*)'
    suffix_pat = r'(?P<suffix>.*)'
    region_coco_name = re.compile(prefix_pat + region_id_pat + suffix_pat + '.kwcoco.zip')

    # Find assocation between src and dst paths based on heuristic region-id
    src_fpaths = util_path.coerce_patterned_paths(config.src_kwcocos)
    dst_fpaths = util_path.coerce_patterned_paths(config.dst_kwcocos)

    region_id_to_src_paths = ub.ddict(list)
    region_id_to_dst_paths = ub.ddict(list)
    for fpath in src_fpaths:
        match = region_coco_name.match(fpath.name)
        region_id = match.groupdict()['region_id']
        region_id_to_src_paths[region_id].append(fpath)

    for fpath in dst_fpaths:
        match = region_coco_name.match(fpath.name)
        region_id = match.groupdict()['region_id']
        region_id_to_dst_paths[region_id].append(fpath)

    # Given the association, the reset of this script is robust.
    assert max(map(len, region_id_to_src_paths.values())) == 1
    assert max(map(len, region_id_to_dst_paths.values())) == 1

    common_region_ids = set(region_id_to_src_paths) & set(region_id_to_dst_paths)

    from geowatch.mlops.pipeline_nodes import ProcessNode

    def submit_job_step(node, depends=None, name=None):
        if config.skip_existing and node.outputs_exist:
            job = None
        else:
            node.cache = config.cache
            job = queue.submit(node.final_command(), depends=depends, name=name)
        return job

    queue = config.create_queue()
    for region_id in common_region_ids:
        src_fpath = region_id_to_src_paths[region_id][0]
        dst_fpath = region_id_to_dst_paths[region_id][0]
        match = region_coco_name.match(dst_fpath.name)
        prefix = match.groupdict()['prefix']
        suffix = match.groupdict()['suffix']
        if suffix == '':
            suffix = '_'
        suffix = suffix + config.new_suffix
        new_fname = prefix + region_id + suffix + '.kwcoco.zip'
        new_fpath = dst_fpath.parent / new_fname

        node = ProcessNode(
            command=ub.codeblock(
                r'''
                python -m geowatch.tasks.cold.transfer_features
                '''),
            in_paths={
                'coco_fpath': src_fpath,
                'combine_fpath': dst_fpath,
            },
            out_paths={
                'new_coco_fpath': new_fpath,
            },
            config={
                'copy_assets': True,
                'io_workers': 4,
            }
        )
        submit_job_step(node, name=f'transfer-cold-{region_id}')

    config.run_queue(queue)


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/geowatch/cli/queue_cli/prepare_cold_transfer.py
        python -m geowatch.cli.queue_cli.prepare_cold_transfer
    """
    main()
