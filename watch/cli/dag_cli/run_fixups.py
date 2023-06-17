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

    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))


def main():
    config = FixupConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))

    from watch.cli.baseline_framework_kwcoco_ingress import baseline_framework_kwcoco_ingress
    from watch.cli.baseline_framework_kwcoco_egress import baseline_framework_kwcoco_egress
    from watch.utils.util_framework import download_region
    from watch.utils.util_framework import AWS_S3_Command
    from watch.utils import util_framework

    # 1. Ingress data
    print("* Running baseline framework kwcoco ingress *")
    ingress_dir = ub.Path('/tmp/ingress')

    baseline_framework_kwcoco_ingress(
        config.input_path,
        ingress_dir,
        config.aws_profile,
        config.dryrun)

    # # 2. Download and prune region file
    print("* Downloading and pruning region file *")
    local_region_path = '/tmp/region.json'

    download_region(
        input_region_path=config.input_region_path,
        output_region_path=local_region_path,
        aws_profile=config.aws_profile,
        strip_nonregions=True,
    )

    dummy_kwcoco_path = ingress_dir / 'sc_fusion_kwcoco.json'

    input_region_dpath = ingress_dir / 'cropped_region_models'
    input_site_dpath = ingress_dir / 'cropped_site_models'

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
    baseline_framework_kwcoco_egress(dummy_kwcoco_path,
                                     local_region_path,
                                     config.output_path,
                                     config.outbucket,
                                     aws_profile=config.aws_profile,
                                     dryrun=False,
                                     newline=False)

    # 6. (Optional) collate TA-2 output
    if config.ta2_s3_collation_bucket is not None:
        print("* Collating TA-2 output")
        aws_cp = AWS_S3_Command('cp', profile=config.aws_profile,
                                dryrun=config.dryrun)
        aws_base_command = aws_cp.finalize()
        util_framework.ta2_collate_output(aws_base_command,
                                          output_region_dpath,
                                          output_site_dpath,
                                          config.ta2_s3_collation_bucket)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/dag_cli/run_fixups.py
    """
    main()
