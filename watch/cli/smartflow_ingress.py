import json
import os
import tempfile

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
            AWS Profile to use for AWS S3 CLI commands
            '''))
    dryrun = scfg.Value(False, isflag=True, group='optional arguments', short_alias=['d'], help='Run AWS CLI commands with --dryrun flag')
    show_progress = scfg.Value(False, isflag=True, group='optional arguments', short_alias=['s'], help='Show progress for AWS CLI commands')
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
                      testing=False,
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

        testing (bool):
            only used in testing. if true, no cp commands are executed.

        dont_error_on_missing_asset (bool):
            if True warn if an asset is missing.
            TODO: variable name is too long and has a double negative.
            maybe rename to "missing_policy" or "ignore_missing"

    Returns:
        Dict[str, str | PathLike]:
            mapping from downloaded assets to their local path

    Example:
        >>> from watch.cli.smartflow_ingress import *  # NOQA
        >>> dpath = ub.Path.appdir('watch/tests/smartflow_ingress').ensuredir()
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
        >>>     'assets': {'asset_file1': {'href': 's3://fake/bucket/my_path.txt'},
        >>>      'asset_dir1': {'href': 's3://fake/bucket/my_dir'}}}]}}
        >>> remote_dpath = (dpath / 'remote').ensuredir()
        >>> input_path = remote_dpath / 'items.jsonl'
        >>> input_path.write_text(json.dumps(demo_stac_content))
        >>> outdir = (dpath / 'local').ensuredir()
        >>> assets = ['asset_file1', 'asset_dir1']
        >>> kwcoco_stac_item_assets = smartflow_ingress(
        >>>     input_path,
        >>>     assets,
        >>>     outdir,
        >>>     dryrun=True, testing=True
        >>> )
        >>> assert kwcoco_stac_item_assets['asset_file1'] == os.fspath(outdir / 'my_path.txt')
        >>> assert kwcoco_stac_item_assets['asset_dir1'] == os.fspath(outdir / 'my_dir')
    """

    from watch.utils.util_framework import AWS_S3_Command
    os.makedirs(outdir, exist_ok=True)

    # TODO: perhaps there is a way to use fsspec to generalize this over
    # different file systems, namely S3, ssh, and local. Being able to run this
    # locally would be nice for testing.

    aws_cp = AWS_S3_Command('cp', profile=aws_profile, dryrun=dryrun,
                            only_show_errors=not show_progress)

    aws_ls = AWS_S3_Command('ls', profile=aws_profile)

    def _load_input(path):
        try:
            with open(path) as f:
                input_json = json.load(f)
            return input_json['stac'].get('features', [])
        # Excepting KeyError here in case of a single line STAC item input
        except (json.decoder.JSONDecodeError, KeyError):
            # Support for simple newline separated STAC items
            with open(path) as f:
                return [json.loads(line) for line in f]

    if input_path.startswith('s3'):
        with tempfile.NamedTemporaryFile() as temporary_file:
            aws_cp.args = [input_path, temporary_file.name]
            aws_cp.run()
            input_stac_items = _load_input(temporary_file.name)
    else:
        input_stac_items = _load_input(input_path)

    # Our baseline KWCOCO egress script should only ever write out a
    # single KWCOCO STAC item
    if len(input_stac_items) != 1:
        raise RuntimeError("Expecting one and only one STAC item from input")
    kwcoco_stac_item = input_stac_items[0]
    kwcoco_stac_item_assets = {
        k: v['href'] for k, v in kwcoco_stac_item['assets'].items()}

    # TODO: can generate a set of download commands that we can execute in
    # parallel
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

        asset_basename = os.path.basename(asset_href)
        asset_outpath = os.path.join(outdir, asset_basename)

        if asset_outpath not in seen:
            aws_ls.args = [asset_href]
            if not dryrun:
                ls_out = aws_ls.run(capture=True)
                # Must correctly set 'recursive' flag for AWS S3 cp calls
                # `aws ls` on a "directory" or really "prefix" of one or more
                # objects in S3 prints "PRE" (indicating it's a prefix)
                if ls_out['out'].strip().startswith('PRE'):
                    aws_cp.update(recursive=True)
                else:
                    aws_cp.update(recursive=False)

            if not testing:
                aws_cp.args = [asset_href, asset_outpath]
                aws_cp.run()
            seen.add(asset_outpath)

        kwcoco_stac_item_assets[asset] = asset_outpath

    # Returns assets (with downloaded asset hrefs updated)
    print('INGRESSED = {}'.format(ub.urepr(kwcoco_stac_item_assets, nl=1)))
    return kwcoco_stac_item_assets


if __name__ == "__main__":
    main()
