import json
import tempfile
import uuid

import ubelt as ub
import scriptconfig as scfg


# FIXME: Looks like this CLI is not functional, which might be fine considering
# this is meant to be used as a library.

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

    from geowatch.geoannots.geomodels import RegionModel
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
        xdoctest -m geowatch.cli.smartflow_egress smartflow_egress

    Example:
        >>> from geowatch.cli.smartflow_egress import *  # NOQA
        >>> from geowatch.geoannots.geomodels import RegionModel
        >>> from os.path import join
        >>> dpath = ub.Path.appdir('geowatch/tests/smartflow_egress').ensuredir()
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

    Ignore:
        >>> # Requires a real S3 bucket
        >>> from geowatch.cli.smartflow_egress import *  # NOQA
        >>> from geowatch.geoannots.geomodels import RegionModel
        >>> from geowatch.utils import util_fsspec
        >>> from os.path import join
        >>> dpath = ub.Path.appdir('geowatch/tests/smartflow_egress').ensuredir()
        >>> local_dpath = (dpath / 'local').ensuredir()
        >>> remote_root = (dpath / 'fake_s3_loc').ensuredir()
        >>> # A REAL AWS PATH WE HAVE ACCESS TO
        >>> outbucket = util_fsspec.S3Path.coerce('s3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/tests/test-egress')
        >>> if 0:
        >>>     outbucket.delete()
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
        >>> outbucket.ls()
        >>> (outbucket / 'my_dir1').ls()
        >>> # Test subsequent ingress
        >>> from geowatch.cli.smartflow_ingress import smartflow_ingress
        >>> in_dpath = ub.Path.appdir('geowatch/tests/smartflow_ingress2').delete().ensuredir()
        >>> input_path = output_path
        >>> assets = ['asset_file1', 'asset_dir1']
        >>> kwcoco_stac_item_assets = smartflow_ingress(
        >>>     input_path,
        >>>     assets,
        >>>     in_dpath,
        >>> )
    """
    # TODO: handle aws_profile.
    from geowatch.utils.util_fsspec import FSPath
    print('--- BEGIN EGRESS ---')
    print(f'outbucket   = {outbucket}')
    print(f'output_path = {output_path}')

    assert aws_profile is None, 'unhandled'
    outbucket = FSPath.coerce(outbucket)

    PRE_DELETE_HACK = 1
    if PRE_DELETE_HACK:
        # HACK: delete everything in the outbucket to prevent conflicting
        # results and ensure the next step always gets exactly this output
        # and nothing more.
        if outbucket.exists():
            print(f'DELETE EXISTING: outbucket={outbucket}')
            outbucket.delete()
        else:
            print('Outbucket doesnt exist yet')

    # Make a temporary output path for a transactional upload.
    tmp_prefix = 'tmp-' + ub.timestamp() + '-' + ub.hash_data(uuid.uuid4())[0:8] + '-'
    tmp_parent = (outbucket.parent / 'tmp').ensuredir()
    tmp_outbucket = tmp_parent / (tmp_prefix + outbucket.name)
    tmp_outbucket.ensuredir()

    # TODO: Can use fsspec to grab multiple files in parallel
    assetnames_and_s3_paths = {}

    items = list(assetnames_and_local_paths.items())
    seen = set()  # Prevent duplicate uploads
    for asset, local_path in ub.ProgIter(items, desc='Egress data', verbose=3):
        local_path = FSPath.coerce(local_path)
        if local_path.startswith('s3'):
            # Assets with paths already on S3 simply pass a reference through
            final_asset_s3_outpath = local_path
            tmp_asset_s3_outpath = None
        else:
            # Otherwise do a copy. Mark the temporary transaction location
            # and the real final location.
            final_asset_s3_outpath = outbucket / local_path.name
            tmp_asset_s3_outpath = tmp_outbucket / local_path.name
            if local_path not in seen:
                fallback_copy(local_path, tmp_asset_s3_outpath)
                # local_path.copy(asset_s3_outpath)
                seen.add(local_path)

        assetnames_and_s3_paths[asset] = {'href': str(final_asset_s3_outpath)}

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

    # Finish transaction
    tmp_outbucket.move(outbucket)

    # Write the final file after the move because it often will write into the
    # final directory.
    with tempfile.NamedTemporaryFile() as temporary_file:
        with open(temporary_file.name, 'w') as f:
            if newline:
                print(te_output, end='', file=f)
            else:
                print(json.dumps(te_output, indent=2), file=f)

        _temp_path = FSPath.coerce(temporary_file.name)
        _output_path = FSPath.coerce(output_path)
        fallback_copy(_temp_path, _output_path)

    print('EGRESSED: {}'.format(ub.urepr(output_stac_items, nl=-1)))
    print('--- FINISH EGRESS ---')
    return te_output


def fallback_copy(local_path, asset_s3_outpath):
    """
    Copying with fsspec alone seems to be causing issues.
    This provides a fallback to a raw S3 command, as well as other verbosity.
    """
    from geowatch.utils import util_fsspec
    from geowatch.utils import util_framework
    assert isinstance(local_path, util_fsspec.LocalPath)

    DO_FALLBACK = 1

    # callback seems to break, not sure why, fixme?
    # from fsspec.callbacks import TqdmCallback
    # callback = TqdmCallback(tqdm_kwargs={"desc": "Copying"})
    if local_path.is_dir() and isinstance(asset_s3_outpath, util_fsspec.S3Path):
        if DO_FALLBACK:
            try:
                local_path.copy(asset_s3_outpath, verbose=3)
            except Exception:
                print('fsspec copy failed, fallback to aws CLI')
                # In the case where we are moving a directory from the local to
                # s3 we *should* just be able to use copy, but because that
                # seems to be breaking, we are falling back on an explicit aws
                # cli command
                profile = asset_s3_outpath.fs.storage_options.get('profile', None)
                aws_kwargs = {}
                if profile is not None:
                    aws_kwargs['profile'] = profile
                aws_cmd = util_framework.AWS_S3_Command(
                    'sync', local_path, asset_s3_outpath, **aws_kwargs)
                aws_cmd.run()
        else:
            local_path.copy(asset_s3_outpath, verbose=3)
            #callback=callback)
    else:
        # In every other case, regular copy is probably fine
        local_path.copy(asset_s3_outpath, verbose=3)
        #callback=callback)


if __name__ == "__main__":
    main()
