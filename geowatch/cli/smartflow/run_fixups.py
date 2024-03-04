#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class FixupConfig(scfg.DataConfig):
    """
    Given the final results of the system, run our validation scripts and any
    final cleanups that need to happen before submitting to T&E validation.
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

    ta2_s3_collation_bucket = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            S3 Location for collated TA-2 output (bucket name should
            include up to eval name)
            '''))

    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))

    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='DEPRECATED.')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))

    input_region_models_asset_name = scfg.Value('cropped_region_models_sc', type=str, required=False, help=ub.paragraph(
            '''
            Which region model assets to ingress and fix up
            '''), alias=['region_models_asset_name'])

    input_site_models_asset_name = scfg.Value('cropped_site_models_sc', type=str, required=False, help=ub.paragraph(
        '''
        Which site model assets to ingress and fix up
        '''), alias=['site_models_asset_name'])

    performer_suffix = scfg.Value('KIT', type=str, required=True, help=ub.paragraph(
            '''
            Performer suffix for output files
            '''))


def main(cmdline=1, **kwargs):
    config = FixupConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    print('config = {}'.format(ub.urepr(config, nl=1, align=':')))

    from geowatch.cli.smartflow_ingress import smartflow_ingress
    from geowatch.cli.smartflow_egress import smartflow_egress
    from geowatch.utils.util_framework import download_region
    from geowatch.utils import util_framework

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = ub.Path('/tmp/ingress')

    input_path = config.input_path
    assets = [
        {'key': config.input_region_models_asset_name},
        {'key': config.input_site_models_asset_name, 'missing_action': 'mkdir'},
    ]
    outdir = ingress_dir
    aws_profile = config.aws_profile
    dryrun = config.dryrun
    # show_progress = False
    # dont_error_on_missing_asset = False

    ingressed_assets = smartflow_ingress(
        input_path,
        assets,
        outdir,
        aws_profile,
        dryrun)

    # # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'

    download_region(
        input_region_path=config.input_region_path,
        output_region_path=local_region_path,
        aws_profile=config.aws_profile,
        strip_nonregions=True,
    )

    dummy_kwcoco_path = ingress_dir / 'dummy.kwcoco.json'
    dummy_kwcoco_path.touch()

    input_region_dpath = ub.Path(ingressed_assets[config.input_region_models_asset_name])
    input_site_dpath = ub.Path(ingressed_assets[config.input_site_models_asset_name])

    output_region_dpath = ingress_dir / 'cropped_region_models_fixed'
    output_site_dpath = ingress_dir / 'cropped_site_models_fixed'

    # Copy the input to the output because we will modify them inplace.
    input_site_dpath.copy(output_site_dpath)
    input_region_dpath.copy(output_region_dpath)

    # Validate and fix all outputs
    print('Fixup and validate outputs')
    util_framework.fixup_and_validate_site_and_region_models(
        region_dpath=output_region_dpath,
        site_dpath=output_site_dpath,
    )

    # 5. Egress (envelop KWCOCO dataset in a STAC item and egress;
    #    will need to recursive copy the kwcoco output directory up to
    #    S3 bucket)
    print("* Egressing KWCOCO dataset and associated STAC item *")
    ingressed_assets['cropped_region_models_fixed'] = output_region_dpath
    ingressed_assets['cropped_site_models_fixed'] = output_site_dpath
    smartflow_egress(ingressed_assets,
                     local_region_path,
                     config.output_path,
                     config.outbucket,
                     aws_profile=config.aws_profile,
                     dryrun=False,
                     newline=False)

    # 6. (Optional) collate TA-2 output
    if config.ta2_s3_collation_bucket is not None:
        print("* Collating TA-2 output")
        util_framework.ta2_collate_output(None,
                                          output_region_dpath,
                                          output_site_dpath,
                                          config.ta2_s3_collation_bucket,
                                          performer_suffix=config.performer_suffix)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/cli/smartflow/run_fixups.py
    """
    main()
