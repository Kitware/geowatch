#!/usr/bin/env python3
"""
See Old Script:
    ~/code/watch/scripts/run_pseudolive_consolidate.py
"""
import os
import subprocess
import ubelt as ub
import scriptconfig as scfg

from geowatch.utils.util_framework import download_region


class PseudoliveConsolidateConfig(scfg.DataConfig):
    """
    Run pseudolive consolidation script for TA-2 region / site model outputs
    """
    region_id = scfg.Value(None, type=str, position=1, required=True, help='Region ID')
    input_region_path = scfg.Value(None, type=str, position=2, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework Region definition JSON
            '''))
    previous_consolidated_output = scfg.Value(None, type=str, position=3, required=True, help=ub.paragraph(
            '''
            S3 path to consolidated regions / sites from previous
            iteration
            '''))
    current_output = scfg.Value(None, type=str, position=4, required=True, help=ub.paragraph(
            '''
            S3 path to regions / sites from current iteration
            '''))
    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))
    outbucket = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            S3 Output directory for STAC item / asset egress
            '''))
    performer_suffix = scfg.Value(None, type=str, short_alias=['s'], help='Performer suffix if present, e.g. KIT')
    iou_threshold = scfg.Value(0.5, type=float, short_alias=['i'], help=ub.paragraph(
            '''
            IOU Threshold for determining duplicates(default: 0.5)
            '''))
    just_deconflict = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Don't copy previous sites, just deconflict current site IDs
            with respect to previous
            '''))


def main():
    config = PseudoliveConsolidateConfig.cli(strict=True)
    print('config = {}'.format(ub.urepr(dict(config), nl=1, align=':')))
    run_pseudolive_consolidate(config)


def run_pseudolive_consolidate(config):

    from geowatch.cli.pseudolive_consolidate import pseudolive_consolidate
    if config.aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', config.aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    local_region_path = ub.Path('/tmp/region.json')
    local_region_path = download_region(config.input_region_path,
                                        local_region_path,
                                        aws_profile=config.aws_profile,
                                        strip_nonregions=True)

    local_previous_dir = os.path.join('/tmp', 'previous_data')
    local_current_dir = os.path.join('/tmp', 'current_data')

    subprocess.run([*aws_base_command, '--recursive',
                    config.previous_consolidated_output, local_previous_dir],
                   check=True)

    # Quickly check and short-circuit if no "previous" data was copied
    # over (e.g. if there was no previous data)
    if len(os.listdir(local_previous_dir)) == 0:
        subprocess.run([*aws_base_command, '--recursive',
                        config.current_output, config.outbucket],
                       check=True)
        return

    subprocess.run([*aws_base_command, '--recursive',
                    config.current_output, local_current_dir],
                   check=True)

    local_consolidated_dir = os.path.join('/tmp', 'consolidated_out')
    pseudolive_consolidate(config.region_id,
                           local_region_path,
                           local_previous_dir,
                           local_current_dir,
                           local_consolidated_dir,
                           config.iou_threshold,
                           performer_suffix=config.performer_suffix,
                           just_deconflict=config.just_deconflict)

    subprocess.run([*aws_base_command, '--recursive',
                    local_consolidated_dir, config.outbucket],
                   check=True)


if __name__ == "__main__":
    main()
