import argparse
import sys
import json
import tempfile
import subprocess
from datetime import datetime
from urllib.parse import urlparse
import glob
import os

from shapely.ops import unary_union
from shapely.geometry import shape, mapping


def main():
    parser = argparse.ArgumentParser(
        description="Add sites to region file and push")

    parser.add_argument('input_region_path',
                        type=str,
                        help="Path to input T&E Baseline Framework Region "
                             "definition JSON")
    parser.add_argument('output_region_path',
                        type=str,
                        help="Output path to T&E Baseline Framework formatted "
                             "Region definition with added sites")
    parser.add_argument('site_models_dir',
                        type=str,
                        help="Directory containing Site models to add to "
                             "region")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("-d", "--dryrun",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with --dryrun flag")

    add_sites_to_region(**vars(parser.parse_args()))

    return 0


def _generate_site_summary(site_path):
    with open(site_path) as f:
        site = json.load(f)

    start_date = None
    end_date = None
    feature_geoms = []
    for feature in site.get('features', ()):
        feature_geoms.append(shape(feature['geometry']))

        date = datetime.fromisoformat(
            feature['properties']['observation_date'])

        if start_date is None or date < start_date:
            start_date = date

        if end_date is None or date > end_date:
            end_date = date

    union_geometry = unary_union(feature_geoms)

    return {"type": "Feature",
            "properties": {
                "type": "site_summary",
                "site_id": site['id'],
                "version": site['version'],
                "mgrs": site['mgrs'],
                "status": "system_proposed",
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "score": site['score'],
                "model_content": "proposed",
                "originator": "kitware"
            },
            "geometry": mapping(union_geometry)}


def add_sites_to_region(input_region_path,
                        output_region_path,
                        site_models_dir,
                        aws_profile=None,
                        dryrun=False):
    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    if urlparse(input_region_path).scheme == 's3':
        with tempfile.NamedTemporaryFile() as temporary_file:
            subprocess.run([*aws_base_command,
                            input_region_path, temporary_file.name],
                           check=True)

            with open(temporary_file.name) as f:
                region = json.load(f)
    else:
        with open(input_region_path) as f:
            region = json.load(f)

    new_site_summary_features =\
        [_generate_site_summary(site_path)
         for site_path in glob.glob(os.path.join(site_models_dir, '*json'))]

    region.setdefault('features', []).extend(new_site_summary_features)

    if urlparse(output_region_path).scheme == 's3':
        with tempfile.NamedTemporaryFile() as temporary_file:
            with open(temporary_file.name, 'w') as f:
                print(json.dumps(region, indent=2), file=f)

                subprocess.run([*aws_base_command,
                                temporary_file.name, output_region_path],
                               check=True)
    else:
        with open(output_region_path, 'w') as f:
            json.dump(region, f, indent=2)

    return region


if __name__ == "__main__":
    sys.exit(main())
