#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub

"""
Notes:

    A quick local test to see if the tensorboard dependencies are acting up.

    # Set this to the docker image you want to test.
    IMAGE=watch:0.7.4-95c6b4b2-strict-pyenv3.11.2-20230623T134259-0400-from-01080c26

    docker run -it $IMAGE python -c "if 1:
        import numpy as np
        import watch
        import ubelt as ub
        from watch.tasks.depth_pcd.model import getModel
        model = getModel()
        expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        model.load_weights(expt_dvc_dpath + '/models/depth_pcd/basicModel2.h5')
        out = model.predict(np.zeros((1,400,400,3)))
        shapes = [o.shape for o in out]
        print('shapes = {}'.format(ub.urepr(shapes, nl=1)))
    "

"""


class DzyneParallelSiteValiConfig(scfg.DataConfig):
    """
    Run DZYNE's parallel site validation framework component

    python ~/code/watch/watch/cli/smartflow/run_dzyne_parallel_site_vali.py
    """
    input_path = scfg.Value(None, type=str, position=1, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework JSON
            '''))

    input_region_path = scfg.Value(None, type=str, position=2, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework Region definition JSON
            '''))

    output_path = scfg.Value(None, type=str, position=3, required=True, help='S3 path for output JSON')

    # depth_model_fpath = scfg.Value("/models/depthPCD/basicModel2.h5", type=str, position=4, required=True, help='path to depth model weights')

    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))

    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')

    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))

    depth_score_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for depth scorer.
            '''))

    depth_filter_config = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Raw json/yaml or a path to a json/yaml file that specifies the
            config for the depth filter.
            '''))


def main():
    config = DzyneParallelSiteValiConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(config, nl=1, align=':')))
    run_dzyne_parallel_site_vali_for_baseline(config)


def run_dzyne_parallel_site_vali_for_baseline(config):
    from watch.cli.smartflow_ingress import smartflow_ingress
    from watch.cli.smartflow_egress import smartflow_egress
    # from watch.cli.concat_kwcoco_videos import concat_kwcoco_datasets
    from watch.utils.util_framework import download_region, determine_region_id
    # from watch.tasks.fusion.predict import predict
    # from kwutil.util_yaml import Yaml

    input_path = config.input_path
    input_region_path = config.input_region_path
    output_path = config.output_path
    outbucket = config.outbucket
    aws_profile = config.aws_profile
    dryrun = config.dryrun

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = ub.Path('/tmp/ingress')
    ingressed_assets = smartflow_ingress(
        input_path,
        ['cropped_kwcoco_for_sv',
         'cropped_kwcoco_for_sv_assets',
         'cropped_site_models_bas',
         'cropped_region_models_bas'],
        ingress_dir,
        aws_profile,
        dryrun)

    # # 2. Download and prune region file
    print("* Downloading and pruning region file *")

    # CHECKME: Is this the region model with site summaries from the tracker?
    local_region_path = '/tmp/region.json'

    download_region(
        input_region_path=input_region_path,
        output_region_path=local_region_path,
        aws_profile=aws_profile,
        strip_nonregions=True,
        ensure_comments=True,
    )

    # Determine the region_id in the region file.
    region_id = determine_region_id(local_region_path)

    ####
    # DEBUGGING:
    # Print info about what version of the code we are running on
    ub.cmd('git log -n 1', verbose=3, cwd='/root/code/watch')

    # 3. Run the Site Validation Filter
    print("* Running the Site Validation Filter *")
    from watch.tasks.depth_pcd import score_tracks
    from watch.tasks.depth_pcd import filter_tracks
    from kwutil.util_yaml import Yaml

    default_score_config = ub.udict({
        'model_fpath': None,
    })
    score_config = (default_score_config
                    | Yaml.coerce(config.depth_score_config or {}))
    if score_config.get('model_fpath', None) is None:
        raise ValueError('Requires model_fpath')

    default_filter_config = ub.udict({
        'threshold': 0.4,
    })
    filter_config = (default_filter_config
                     | Yaml.coerce(config.depth_filter_config or {}))

    # 3.3 Run DZYNE depth_pcd
    print("* Running DZYNE depth_pcd *")

    # TODO: The input / output site and region paths should be specified as
    # parameters passed to us by the DAG.
    input_kwcoco_fpath = ingressed_assets['cropped_kwcoco_for_sv']
    input_sites_dpath = ingressed_assets['cropped_site_models_bas']
    input_region_dpath = ingressed_assets['cropped_region_models_bas']
    input_region_fpath = ub.Path(input_region_dpath) / f'{region_id}.geojson'
    # input_region_fpath = local_region_path  # is this right?

    scored_kwcoco_fpath = ingress_dir / "poly_depth_scored.kwcoco.zip"

    output_site_manifest_fpath = ingress_dir / "depth_filtered_sites_manifest.json"
    output_sites_dpath = ingress_dir / "depth_filtered_sites"
    output_region_dpath = ingress_dir / "depth_filtered_regions"
    output_region_fpath = output_region_dpath / f'{region_id}.geojson'

    score_tracks.main(
        cmdline=0,

        **score_config,

        # Should be the SV-cropped kwcoco file that contains start and ending
        # high resolution images where videos correspond to proposed sites for
        # this region.
        input_kwcoco=input_kwcoco_fpath,

        # Should be the region models containing the current site summaries
        # from the previous step.
        input_region=input_region_fpath,

        # This is a kwcoco file used internally in this step where scores
        # are assigned to each track. The next step will use this.
        out_kwcoco=scored_kwcoco_fpath,
    )
    filter_tracks.main(
        cmdline=0,

        **filter_config,

        # The kwcoco file contining depth scores that this step will use to
        # filter the input sites / site summaries.
        input_kwcoco=scored_kwcoco_fpath,

        # Should be the region models containing the current site summaries
        # from the previous step.
        input_region=input_region_fpath,

        # Should be the folder containing all of the sites corresponding to the
        # sites in the input_region
        input_sites=input_sites_dpath,

        # The output region model to be used by the next step
        output_region_fpath=output_region_fpath,

        # The output directory of corresponding site models that should be used by the next step
        output_sites_dpath=output_sites_dpath,

        # A single file that registers all of the sites writen to the output
        # site directory.
        output_site_manifest_fpath=output_site_manifest_fpath,
    )

    # Validate and fix all outputs
    from watch.utils import util_framework
    util_framework.fixup_and_validate_site_and_region_models(
        region_dpath=output_region_fpath.parent,
        site_dpath=output_sites_dpath,
    )

    # 4. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    ingressed_assets['depth_filtered_sites'] = output_sites_dpath
    ingressed_assets['depth_filtered_regions'] = output_region_dpath
    smartflow_egress(ingressed_assets,
                     local_region_path,
                     output_path,
                     outbucket,
                     aws_profile=aws_profile,
                     dryrun=dryrun,
                     newline=False)


if __name__ == "__main__":
    main()