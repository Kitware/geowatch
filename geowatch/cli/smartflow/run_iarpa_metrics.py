#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


__ignore__ = r"""


LOCAL_WORK_DPATH=$HOME/temp/debug_smartflow_metrics/ingress
mkdir -p $LOCAL_WORK_DPATH
cd $LOCAL_WORK_DPATH
docker run \
    --runtime=nvidia \
    --volume "$LOCAL_WORK_DPATH":/tmp/ingress \
    --volume $HOME/.aws:/root/.aws:ro \
    --volume "$HOME/code":/extern_code:ro \
    --volume "$HOME/data":/extern_data:ro \
    --volume "$HOME"/.cache/pip:/pip_cache \
    --env AWS_PROFILE=iarpa \
    -it registry.smartgitlab.com/kitware/geowatch:0.16.2-17a6765ff-strict-pyenv3.11.2-20240410T192350-0400-from-0da55667 bash



from geowatch.cli.smartflow.run_iarpa_metrics import *  # NOQA"
config = RunIARPAMetricsCLI(**{
    'region_id': 'KR_R002',
    'input_region_path': 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v193/batch/kit/KR_R002/split_input/52SDG98/region_models/KR_R002.geojson',
    'true_annot_dpath': 's3://smart-imagery/annotations',
    'pred_site_dpath': 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v193/batch/kit/KR_R002/consolidated_output_sv/site_models',
    'outbucket': 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v193/batch/kit/KR_R002/metrics_sv',
    'output_path': 's3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval21_batch_v193/batch/kit/KR_R002/metrics_sv/items.jsonl',
    'aws_profile': None,
})

cls = RunIARPAMetricsCLI
cmdline = 0
kwargs = dict(config)
"""


class RunIARPAMetricsCLI(scfg.DataConfig):
    """
    This is an entrypoint to run the IARPA metrics code with Kitware
    modifications, this should be used for debugging and internal metrics, the
    official IARPA image should be used for final scores, and ideally these
    will match results produced here.

    Note:
        This currently runs based on paths and not STAC catalogs, which is an
        issue that would be nice to fix.
    """
    region_id = None
    input_region_path = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        Path to input T&E Baseline Framework Region definition JSON
        '''))
    true_annot_dpath = None
    pred_site_dpath = None
    outbucket = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        S3 Output directory for STAC item / asset egress
        '''))

    output_path = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        Path to the STAC items that register the outputs of this stage.
        This is usually an S3 Path.
        '''), alias=['output_stac_path'])

    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
        '''
        AWS Profile to use for AWS S3 CLI commands
        '''))

    # input_path = scfg.Value(None, type=str, position=1, required=True, help=ub.paragraph(
    #         '''
    #         Path to input T&E Baseline Framework JSON
    #         '''))
    # input_region_models_asset_name = scfg.Value('cropped_region_models_sc', type=str, required=False, help=ub.paragraph(
    #         '''
    #         Which region model assets to ingress and fix up
    #         '''), alias=['region_models_asset_name'])
    # input_site_models_asset_name = scfg.Value('cropped_site_models_sc', type=str, required=False, help=ub.paragraph(
    #     '''
    #     Which site model assets to ingress and fix up
    #     '''), alias=['site_models_asset_name'])

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from geowatch.cli.smartflow.run_metrics import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = RunIARPAMetricsCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        print('config = ' + ub.urepr(config, nl=1))

        from geowatch.utils.util_framework import NodeStateDebugger
        node_state = NodeStateDebugger()
        node_state.print_environment()
        node_state.print_local_invocation(config)

        from geowatch.utils.util_framework import download_region
        from geowatch.mlops import smart_pipeline
        from geowatch.utils.util_fsspec import FSPath

        # 1. Ingress data
        print("* Running baseline framework kwcoco ingress *")
        ingress_dir = ub.Path('/tmp/ingress')

        eval_dpath = (ingress_dir / 'metrics_output').ensuredir()
        viz_output_dpath = (eval_dpath / 'region_viz_overall').ensuredir()

        USE_NON_STAC_PATHS = True
        if USE_NON_STAC_PATHS:
            # FIXME: would be better if we conformed to the same STAC-in
            # STAC-out paradigm everything else uses, but for now
            # use raw S3 paths.
            remote_true_annot_dpath = FSPath.coerce(config.true_annot_dpath)
            remote_true_region_dpath = remote_true_annot_dpath / 'region_models'
            remote_true_site_dpath = remote_true_annot_dpath / 'site_models'

            remote_pred_site_dpath = FSPath.coerce(config.pred_site_dpath)

            # Select the truth sites to pull down
            remote_true_site_fpaths = [p for p in remote_true_site_dpath.ls() if p.name.startswith(config.region_id)]
            remote_true_region_fpaths = [p for p in remote_true_region_dpath.ls() if p.name.startswith(config.region_id)]

            true_annot_dpath = (ingress_dir / 'truth').ensuredir()
            true_region_dpath = (true_annot_dpath / 'region_models').ensuredir()
            true_site_dpath = (true_annot_dpath / 'site_models').ensuredir()

            # Copy predictions to local node
            pred_annot_dpath = (ingress_dir / 'pred').ensuredir()
            pred_site_dpath = FSPath.coerce(pred_annot_dpath / 'site_models')
            remote_pred_site_dpath.copy(pred_site_dpath, verbose=3)

            # We dont need the region model for scoring, but it helps for
            # visualization and ensuring our approch is consistent
            remote_pred_region_dpath = remote_pred_site_dpath.parent / 'region_models'
            pred_region_dpath = FSPath.coerce(pred_annot_dpath / 'region_models')
            remote_pred_region_dpath.copy(pred_region_dpath)

            # Copy truth to local node
            for fpath in ub.ProgIter(remote_true_site_fpaths, desc='pull site truth', verbose=3):
                fpath.copy(true_site_dpath / fpath.name)
            for fpath in ub.ProgIter(remote_true_region_fpaths, desc='pull region truth', verbose=3):
                fpath.copy(true_region_dpath / fpath.name)

        # 2. Download and prune region file
        print("* Downloading and pruning region file *")
        local_region_path = '/tmp/region.json'
        download_region(
            input_region_path=config.input_region_path,
            output_region_path=local_region_path,
            aws_profile=config.aws_profile,
            strip_nonregions=True,
        )

        node_state.print_current_state(ingress_dir)

        DRAW_SANITY_CHECK = True
        if DRAW_SANITY_CHECK:
            print('Draw predictions and truth before we do eval as a sanity check')
            command = ub.paragraph(
                f'''
                geowatch draw_region {true_site_dpath}
                    --extra_header "True Sites"
                    --fpath "{viz_output_dpath}"/true_site_viz.png
                ''')
            ub.cmd(command, shell=True, verbose=3)

            command = ub.paragraph(
                f'''
                geowatch draw_region {true_region_dpath}
                    --extra_header "True Region"
                    --fpath "{viz_output_dpath}"/true_region_viz.png
                ''')
            ub.cmd(command, shell=True, verbose=3)

            command = ub.paragraph(
                f'''
                geowatch draw_region {pred_site_dpath}
                    --extra_header "Pred Sites"
                    --fpath "{viz_output_dpath}"/pred_site_viz.png
                ''')
            ub.cmd(command, shell=True, verbose=3)

            command = ub.paragraph(
                f'''
                geowatch draw_region {pred_region_dpath}
                    --extra_header "Pred Regions"
                    --fpath "{viz_output_dpath}"/pred_region_viz.png
                ''')
            ub.cmd(command, shell=True, verbose=3)

        smart_pipeline.PolygonEvaluation.name = 'poly_eval'
        eval_node = smart_pipeline.PolygonEvaluation()
        eval_fpath = eval_dpath / 'poly_eval.json'

        eval_node.configure({
            'sites_fpath': pred_site_dpath,  # note this is a bad name, should be "pred_sites"
            'true_site_dpath': true_site_dpath,
            'true_region_dpath': true_region_dpath,
            'eval_dpath': eval_dpath,
            'eval_fpath': eval_fpath,
            'enable_viz': True,  # todo: expose
        })
        command = eval_node.command().rstrip('\\')
        print(command)
        ub.cmd(command, check=True, verbose=3, system=True)

        node_state.print_current_state(ingress_dir)

        node_state.print_directory_contents(eval_dpath)

        # Change viz symlinks to real files
        if True:
            for path in viz_output_dpath.glob('*'):
                if path.is_symlink():
                    real_path = path.resolve()
                    assert not real_path.is_symlink()
                    path.unlink()
                    real_path.copy(path)

        if True:
            confusion_out_dpath = eval_dpath / 'confusion_analysis'
            region_id = config.region_id
            bas_metric_dpath = (eval_dpath / region_id / 'overall/bas')
            src_kwcoco_fpath = None  # TODO: get imagery
            command = ub.codeblock(
                fr'''
                python -m geowatch.mlops.confusor_analysis \
                    --src_kwcoco="{src_kwcoco_fpath}" \
                    --bas_metric_dpath="{bas_metric_dpath}" \
                    --pred_sites="{pred_site_dpath}" \
                    --out_dpath={confusion_out_dpath} \
                    --true_region_dpath="{true_region_dpath}" \
                    --true_site_dpath="{true_site_dpath}" \
                    --region_id="{region_id}" \
                    --viz_sites=0 \
                    --reload=0
                ''')
            ub.cmd(command, verbose=3, shell=1)

            confusion_group_dpath = confusion_out_dpath / 'confusion_groups'
            confusion_region_fpaths = list(confusion_group_dpath.glob('*.geojson'))
            for fpath in confusion_region_fpaths:
                command = ub.paragraph(
                    f'''
                    geowatch draw_region {fpath}
                        --extra_header "{fpath.name}"
                        --fpath "{fpath}.png"
                    ''')
                ub.cmd(command, shell=True, verbose=3)

        assets_to_egress = {
            'eval_dpath': eval_dpath,
            'eval_fpath': eval_fpath,
        }
        from geowatch.cli.smartflow_egress import smartflow_egress
        smartflow_egress(assets_to_egress,
                         local_region_path,
                         config.output_path,
                         config.outbucket,
                         aws_profile=config.aws_profile,
                         # dryrun=config.dryrun,
                         # newline=config.newline
                         )


__cli__ = RunIARPAMetricsCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/geowatch/cli/smartflow/run_metrics.py
        python -m geowatch.cli.smartflow.run_metrics
    """
    main()
