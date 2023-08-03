import json
import tempfile
import uuid

import ubelt as ub
import scriptconfig as scfg


# FIXME: Looks like this CLI is not functional.

class SmartflowEgressConfig(scfg.DataConfig):
    """
    Egress KWCOCO data to T&E baseline framework structure
    """
    input_path = scfg.Value(None, type=str, position=1, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework JSON
            '''))
    assets = scfg.Value(None, type=str, position=2, required=True, help='Assets to upload', nargs='+')
    outdir = scfg.Value(None, type=str, required=True, short_alias=['o'], help=ub.paragraph(
            '''
            Output directory for ingressed assets an output STAC Catalog
            '''))
    aws_profile = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            AWS Profile to use. UNUSED. Hook up to fsspec if needed.
            '''))
    dryrun = scfg.Value(False, isflag=True, short_alias=['d'], help='UNUSED. DEPRECATED')
    show_progress = scfg.Value(False, isflag=True, short_alias=['s'], help='UNUSED. DEPRECATED')


def main():
    smartflow_egress_with_arg_processing(**SmartflowEgressConfig.cli())


def _build_stac_item(region_path,
                     assetnames_and_s3_paths):
    with open(region_path) as f:
        data = json.load(f)

    from watch.geoannots.geomodels import RegionModel
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
            'geometry': shapely.geometry.mapping(region_geometry),
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
    """
    Uploads specified assets to S3 with a STAC manifest.

    Args:
        assetnames_and_local_paths (Dict):
            Mapping from an asset name to the local path to upload. The asset
            name will be indexable in the uploaded STAC item. Any local path
            specified multiple times will only be uploaded once, but multiple
            STAC assets will be associated with it.

        region_path (str | PathLike):
            local path to the region file associated with a processing node

        output_path (str):
            The path in the s3 bucket that the stac item will be uploaded to.

        outbucket (str):
            The s3 bucket that assets will be uploaded to.

        aws_profile (str | None): aws cp argument

        newline (bool): controls formatting of output stac item

    Returns:
        Dict: the new STAC item

    CommandLine:
        xdoctest -m watch.cli.smartflow_egress smartflow_egress

    Example:
        >>> from watch.cli.smartflow_egress import *  # NOQA
        >>> from watch.geoannots.geomodels import RegionModel
        >>> from os.path import join
        >>> dpath = ub.Path.appdir('watch/tests/smartflow_egress').ensuredir()
        >>> local_dpath = (dpath / 'local').ensuredir()
        >>> remote_root = (dpath / 'fake_s3_loc').ensuredir()
        >>> #outbucket = 's3://fake/bucket'
        >>> outbucket = remote_root
        >>> output_path = join(outbucket, 'items.jsonl')
        >>> region = RegionModel.random()
        >>> region_path = dpath / 'demo_region.geojson'
        >>> region_path.write_text(region.dumps())
        >>> assetnames_and_local_paths = {
        >>>     'asset_file1': dpath / 'my_path1.txt',
        >>>     'asset_file2': dpath / 'my_path2.txt',
        >>>     'asset_file_reference': dpath / 'my_path1.txt',
        >>>     'asset_dir1': dpath / 'my_dir1',
        >>> }
        >>> # Generate local data we will pretend to egress
        >>> assetnames_and_local_paths['asset_file1'].write_text('foobar1')
        >>> assetnames_and_local_paths['asset_file2'].write_text('foobar2')
        >>> assetnames_and_local_paths['asset_dir1'].ensuredir()
        >>> (assetnames_and_local_paths['asset_dir1'] / 'data1').write_text('data1')
        >>> (assetnames_and_local_paths['asset_dir1'] / 'data1').write_text('data2')
        >>> te_output = smartflow_egress(
        >>>     assetnames_and_local_paths,
        >>>     region_path,
        >>>     output_path,
        >>>     outbucket,
        >>>     newline=False,
        >>> )
    """
    # TODO: handle aws_profile.
    from watch.utils.util_fsspec import FSPath

    assert aws_profile is None, 'unhandled'
    outbucket = FSPath.coerce(outbucket)

    # TODO: Can use fsspec to grab multiple files in parallel
    seen = set()  # Prevent duplicate uploads
    assetnames_and_s3_paths = {}
    for asset, local_path in assetnames_and_local_paths.items():
        # Assets with paths already on S3 simply pass a reference through
        local_path = FSPath.coerce(local_path)
        if local_path.startswith('s3'):
            asset_s3_outpath = local_path
        else:
            asset_s3_outpath = outbucket / local_path.name
            if local_path not in seen:
                local_path.copy(asset_s3_outpath)
                seen.add(local_path)

        assetnames_and_s3_paths[asset] = {'href': str(asset_s3_outpath)}

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

        _temp_path = FSPath.coerce(temporary_file.name)
        _output_path = FSPath.coerce(output_path)
        _temp_path.copy(_output_path)

    print('EGRESSED: {}'.format(ub.urepr(te_output, nl=-1)))
    return te_output


if __name__ == "__main__":
    main()
