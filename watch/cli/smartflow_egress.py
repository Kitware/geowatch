import argparse
import sys
import json
import tempfile
from os.path import join, dirname, basename, isdir
import uuid

from shapely.geometry import shape


def main():
    parser = argparse.ArgumentParser(
        description="Egress KWCOCO data to T&E baseline framework structure")

    parser.add_argument('egress_item_name',
                        type=str,
                        help="Path to input T&E Baseline Framework Region "
                             "definition JSON")
    parser.add_argument('assetnames_and_paths',
                        type=str,
                        nargs='+',
                        help="Assets to egress in the form of "
                             "<asset_name>:<local_asset_path>")
    parser.add_argument('region_path',
                        type=str,
                        help="Path to input T&E Baseline Framework Region "
                             "definition JSON")
    parser.add_argument('output_path',
                        type=str,
                        help="S3 path for output JSON")
    parser.add_argument("-o", "--outbucket",
                        type=str,
                        required=True,
                        help="S3 Output directory for STAC item / asset "
                             "egress")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("-d", "--dryrun",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with --dryrun flag")
    parser.add_argument("-n", "--newline",
                        action='store_true',
                        default=False,
                        help="Output as simple newline separated STAC items")
    parser.add_argument('-s', '--show-progress',
                        action='store_true',
                        default=False,
                        help='Show progress for AWS CLI commands')

    smartflow_egress_with_arg_processing(**vars(parser.parse_args()))

    return 0


def _build_stac_item(region_path,
                     assetnames_and_s3_paths):
    with open(region_path) as f:
        region = json.load(f)

    region_features = []
    for feature in region.get('features', ()):
        if ('properties' in feature
           and feature['properties'].get('type') == 'region'):
            region_features.append(feature)

    # WARNING: Big assumption here that we only ever have a single
    # feature with type 'region' in the region file
    if len(region_features) != 1:
        raise RuntimeError("Expecting one and only one feature of type "
                           "'region' in region file")
    region_feature = region_features[0]

    region_geometry = region_feature['geometry']
    region_bbox = list(shape(region_geometry).bounds)

    return {'type': 'Feature',
            'stac_version': '1.0.0',
            'stac_extensions': [],
            'id': uuid.uuid4().hex,
            'geometry': region_geometry,
            'bbox': region_bbox,
            'properties': {},
            'assets': assetnames_and_s3_paths}


def smartflow_egress_with_arg_processing(
        assetnames_and_paths,
        region_path,
        output_path,
        outbucket,
        aws_profile=None,
        dryrun=False,
        newline=False,
        show_progress=False):
    assetnames_and_local_paths = {}
    for assetname_and_path in assetnames_and_paths:
        asset, local_path = assetname_and_path.split(':')

        assetnames_and_local_paths[asset] = local_path

    return smartflow_egress(assetnames_and_local_paths,
                            region_path,
                            output_path,
                            outbucket,
                            aws_profile=aws_profile,
                            dryrun=dryrun,
                            newline=newline,
                            show_progress=show_progress)


def smartflow_egress(assetnames_and_local_paths,
                     region_path,
                     output_path,
                     outbucket,
                     aws_profile=None,
                     dryrun=False,
                     newline=False,
                     show_progress=False):
    from watch.utils.util_framework import AWS_S3_Command
    aws_cp = AWS_S3_Command('cp')
    aws_cp.update(
        profile=aws_profile,
        dryrun=dryrun,
        only_show_errors=not show_progress,
    )

    assetnames_and_s3_paths = {}
    for asset, local_path in assetnames_and_local_paths.items():
        # Passing in assets with paths already on S3 simply passes
        # them through
        if local_path.startswith('s3'):
            asset_s3_outpath = local_path
        else:
            asset_s3_outpath = join(outbucket, basename(local_path))

            if isdir(local_path):
                aws_cp.update(recursive=True)
            else:
                aws_cp.update(recursive=False)

            aws_cp.args = [local_path, asset_s3_outpath]
            aws_cp.run()

        assetnames_and_s3_paths[asset] = {'href': asset_s3_outpath}

    output_stac_item = _build_stac_item(region_path,
                                        assetnames_and_s3_paths)

    # Only have the one KWCOCO dataset STAC item in this case
    output_stac_items = [output_stac_item]
    if newline:
        te_output = '\n'.join((json.dumps(item) for item in output_stac_items))
    else:
        te_output = {'raw_images': [],
                     'stac': {
                         'type': 'FeatureCollection',
                         'features': output_stac_items}}

    with tempfile.NamedTemporaryFile() as temporary_file:
        with open(temporary_file.name, 'w') as f:
            if newline:
                print(te_output, end='', file=f)
            else:
                print(json.dumps(te_output, indent=2), file=f)

        aws_cp.update(recursive=False)
        aws_cp.args = [temporary_file.name, output_path]
        aws_cp.run()

    return te_output


if __name__ == "__main__":
    sys.exit(main())
