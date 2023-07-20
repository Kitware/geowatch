import argparse
import json
import tempfile
from os.path import join, basename, isdir
import uuid

import ubelt as ub
import scriptconfig as scfg


class SmartflowEgressConfig(scfg.DataConfig):
    """
    Egress KWCOCO data to T&E baseline framework structure
    """
    input_path = scfg.Value(None, type=str, position=1, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework JSON
            '''))
    assets = scfg.Value(None, type=str, position=2, required=True, help='Assets to download', nargs='+')
    outdir = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            Output directory for ingressed assets an output STAC Catalog
            '''))
    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands
            '''))
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')
    show_progress = scfg.Value(False, isflag=True, short_alias=['s'], help='Show progress for AWS CLI commands')
    dont_error_on_missing_asset = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Don't raise error on missing asset, just warn
            '''))


def main():
    smartflow_egress_with_arg_processing(**SmartflowEgressConfig.cli())


def _build_stac_item(region_path,
                     assetnames_and_s3_paths):
    with open(region_path) as f:
        region = json.load(f)

    from watch.geoannots.geomodels import RegionModel

    import json
    with open(region_path) as f:
        data = json.load(f)

    region = RegionModel(**data)

    # These are fast checks that include the assertion that there is only one
    # header (i.e. type=region) feature.
    region._validate_quick_checks()

    import shapely
    region_geometry: shapely.geometry.polygon.Polygon = region.geometry
    region_bbox = list(region_geometry.bounds)

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
    main()
