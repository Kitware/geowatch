import json
import os
import ubelt as ub
import scriptconfig as scfg


class SmartflowIngressConfig(scfg.DataConfig):
    """
    Ingress KWCOCO data to T&E baseline framework structure
    """
    input_path = scfg.Value(None, type=str, position=1, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework JSON
            '''))
    assets = scfg.Value(None, type=str, position=2, required=True, help='Names of assets to download', nargs='+')
    outdir = scfg.Value(None, type=str, group='optional arguments', required=True, short_alias=['o'], help=ub.paragraph(
            '''
            Output directory for ingressed assets an output STAC Catalog
            '''))
    aws_profile = scfg.Value(None, type=str, group='optional arguments', help=ub.paragraph(
            '''
            AWS Profile to use for AWS S3 CLI commands. UNUSED. Hook up to fsspec if needed.
            '''))
    dryrun = scfg.Value(False, isflag=True, group='optional arguments', short_alias=['d'], help='UNUSED. DEPRECATED')
    show_progress = scfg.Value(False, isflag=True, group='optional arguments', short_alias=['s'], help='UNUSED. DEPRECATED')
    dont_error_on_missing_asset = scfg.Value(False, isflag=True, group='optional arguments', help=ub.paragraph(
            '''
            Don't raise error on missing asset, just warn
            '''))


def main():
    smartflow_ingress(**SmartflowIngressConfig.cli())


def smartflow_ingress(input_path,
                      assets,
                      outdir,
                      aws_profile=None,
                      dryrun=False,
                      show_progress=False,
                      dont_error_on_missing_asset=False):
    """
    Downloads a STAC manifest and select items within it.

    Args:
        input_path (str):
            The path in the s3 bucket that the STAC item will be downloaded from.

        assets (List[str]):
            A List of keys into the stac item assets that we will download.

        outdir (str | PathLike):
            local path to download to.

        aws_profile (str | None): aws cp argument

        dryrun (bool): aws cp argument

        show_progress (bool): aws cp argument

        dont_error_on_missing_asset (bool):
            if True warn if an asset is missing.
            TODO: variable name is too long and has a double negative.
            maybe rename to "missing_policy" or "ignore_missing"

    Returns:
        Dict[str, str | PathLike]:
            mapping from downloaded assets to their local path

    Example:
        >>> from watch.cli.smartflow_ingress import *  # NOQA
        >>> dpath = ub.Path.appdir('watch/tests/smartflow_ingress/dst').ensuredir()
        >>> fake_remote = ub.Path.appdir('watch/tests/smartflow_ingress/fake_remote').ensuredir()
        >>> fake_fpath = fake_remote / 'my_path.txt'
        >>> fake_fpath.write_text('foobar')
        >>> fake_dpath = (fake_remote / 'my_dir').ensuredir()
        >>> (fake_dpath / 'content1').touch()
        >>> (fake_dpath / 'content2').touch()
        >>> (fake_dpath / 'subdir1').ensuredir()
        >>> (fake_dpath / 'subdir1/subcontent1').touch()
        >>> (fake_dpath / 'subdir1/subcontent2').touch()
        >>> (fake_dpath / 'subdir1/subsubdir').ensuredir()
        >>> # Save this dummy stac item locally
        >>> # In practice we download it, but we are using dry run mode
        >>> # so we cant do that here.
        >>> demo_stac_content = {'raw_images': [],
        >>>  'stac': {'type': 'FeatureCollection',
        >>>   'features': [{'type': 'Feature',
        >>>     'stac_version': '1.0.0',
        >>>     'stac_extensions': [],
        >>>     'id': '66d3e2f605a44aa8b7bacc6ce7e96b9a',
        >>>     'geometry': {'type': 'Polygon',
        >>>      'coordinates': (((-109.56, 44.56),
        >>>        (-109.57, 44.55),
        >>>        (-109.53, 44.56),
        >>>        (-109.56, 44.56)),)},
        >>>     'bbox': [-109.57, 44.52, -109.51, 44.56],
        >>>     'properties': {},
        >>>     'assets': {'asset_file1': {'href': str(fake_fpath)},
        >>>      'asset_dir1': {'href': str(fake_dpath)}}}]}}
        >>> remote_dpath = (dpath / 'remote').ensuredir()
        >>> input_path = remote_dpath / 'items.jsonl'
        >>> input_path.write_text(json.dumps(demo_stac_content))
        >>> outdir = (dpath / 'local').ensuredir()
        >>> assets = ['asset_file1', 'asset_dir1']
        >>> kwcoco_stac_item_assets = smartflow_ingress(
        >>>     input_path,
        >>>     assets,
        >>>     outdir,
        >>> )
        >>> assert kwcoco_stac_item_assets['asset_file1'] == os.fspath(outdir / 'my_path.txt')
        >>> assert kwcoco_stac_item_assets['asset_dir1'] == os.fspath(outdir / 'my_dir')
        >>> assert len(ub.Path(kwcoco_stac_item_assets['asset_dir1']).ls()) > 0
        >>> assert ub.Path(kwcoco_stac_item_assets['asset_file1']).exists()
    """
    os.makedirs(outdir, exist_ok=True)

    assert aws_profile is None, 'unhandled'
    from watch.utils.util_fsspec import FSPath
    input_path = FSPath.coerce(input_path)

    def _loads_input(text):
        try:
            input_json = json.loads(text)
            return input_json['stac'].get('features', [])
        # Excepting KeyError here in case of a single line STAC item input
        except (json.decoder.JSONDecodeError, KeyError):
            # Support for simple newline separated STAC items
            lines = text.split('\n')
            return [json.loads(line) for line in lines]

    input_text = input_path.read_text()
    input_stac_items = _loads_input(input_text)

    # Our baseline KWCOCO egress script should only ever write out a
    # single KWCOCO STAC item
    if len(input_stac_items) != 1:
        raise RuntimeError("Expecting one and only one STAC item from input")
    kwcoco_stac_item = input_stac_items[0]
    kwcoco_stac_item_assets = {
        k: v['href'] for k, v in kwcoco_stac_item['assets'].items()}

    # TODO: can use fsspec to handle multiple downloads in parallel.
    seen = set()  # Prevent duplicate downloads
    for asset in assets:
        try:
            asset_href = kwcoco_stac_item_assets[asset]
        except KeyError:
            missing_asset_str = (
                f"Expecting asset named {asset!r} in input KWCOCO STAC item"
            )
            if dont_error_on_missing_asset:
                print(f"* Warning: {missing_asset_str!r}")
            else:
                raise RuntimeError(missing_asset_str)  # noqa

        outdir = FSPath.coerce(outdir)
        asset_href = FSPath.coerce(asset_href)
        asset_outpath = outdir / asset_href.name
        if asset_outpath not in seen:
            asset_href.copy(asset_outpath)
        seen.add(asset_outpath)

        kwcoco_stac_item_assets[asset] = str(asset_outpath)

    # Returns assets (with downloaded asset hrefs updated)
    print('INGRESSED = {}'.format(ub.urepr(kwcoco_stac_item_assets, nl=1)))
    return kwcoco_stac_item_assets


if __name__ == "__main__":
    main()
