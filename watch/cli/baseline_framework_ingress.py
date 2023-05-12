import sys
import os
import scriptconfig as scfg
import ubelt as ub


SENTINEL_PLATFORMS = {'sentinel-2b', 'sentinel-2a'}


class BaselineFrameworkIngressConfig(scfg.DataConfig):
    """
    Ingress data from T&E baseline framework input file. The output will be stored as a json catalog
    """
    input_path = scfg.Value(None, type=str, position=1, required=True, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework JSON
            '''))
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
    requester_pays = scfg.Value(False, isflag=True, short_alias=['r'], help=ub.paragraph(
            '''
            Run AWS CLI commands with `--requestor_payer requester` flag
            '''))
    jobs = scfg.Value(1, type=str, short_alias=['j'], help='Number of jobs to run in parallel')
    virtual = scfg.Value(False, isflag=True, help=ub.paragraph(
            '''
            Replace asset hrefs with GDAL Virtual File System links
            '''))
    catalog_fpath = scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Name of the ouptut catalog.
            Defaults to <outdir>/catalog.json
            '''))
    relative = scfg.Value(False, isflag=True, help='if true use relative paths')

    def __post_init__(self):
        # super().__post_init__()
        if self.catalog_fpath is None and self.outdir is not None:
            self.catalog_fpath = os.path.join(self.outdir, 'catalog.json')


def main():
    config = BaselineFrameworkIngressConfig.cli(strict=True)
    import rich
    rich.print(ub.urepr(config))
    baseline_framework_ingress(**config)


def baseline_framework_ingress(input_path,
                               outdir,
                               catalog_fpath=None,
                               aws_profile=None,
                               dryrun=False,
                               show_progress=False,
                               requester_pays=False,
                               relative=False,
                               jobs=1,
                               virtual=False):

    from watch.utils import util_parallel
    from watch.utils import util_progress
    import rich
    import pystac
    import traceback
    workers = util_parallel.coerce_num_workers(jobs)
    print(f'Runing baseline_framework_ingress with workers={workers}')

    os.makedirs(outdir, exist_ok=True)

    if relative:
        catalog_type = pystac.CatalogType.RELATIVE_PUBLISHED
    else:
        catalog_type = pystac.CatalogType.ABSOLUTE_PUBLISHED

    if catalog_fpath is None:
        catalog_fpath = os.path.join(outdir, 'catalog.json')
    catalog = pystac.Catalog('Baseline Framework ingress catalog',
                             'STAC catalog of SMART search results',
                             href=catalog_fpath, catalog_type=catalog_type)

    catalog.set_root(catalog)

    if relative:
        catalog.make_all_asset_hrefs_relative()

    if aws_profile is not None:
        aws_base_command = ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    if not show_progress:
        aws_base_command.append('--no-progress')

    if requester_pays:
        aws_base_command.extend(['--request-payer', 'requester'])

    input_stac_items = load_input_stac_items(input_path, aws_base_command)

    print(f'Loaded {len(input_stac_items)} stac items')

    ingress_kw = {
        'outdir': outdir,
        'aws_base_command': aws_base_command,
        'dryrun': dryrun,
        'relative': relative,
        'virtual': virtual,
    }

    pool = ub.JobPool(mode='thread' if workers > 1 else 'serial',
                      max_workers=workers)
    pman = util_progress.ProgressManager(backend='progiter')
    with pman, pool:
        """
        DEVELOPER NOTE:
            There is something that can cause a lockup here. To reproduce
            first ensure that the outdir is cleared, so no caching happens.
            The failure seems to happen when the mode is process. Using thread
            or serial seems fine.
        """
        for feature in pman.progiter(input_stac_items, desc='submit ingress jobs'):
            pool.submit(ingress_item, feature, **ingress_kw)

        for job in pman.progiter(pool.as_completed(), total=len(pool), desc='ingress items'):
            try:
                mapped_item = job.result()
            except Exception:
                rich.print("[yellow]WARNING: Exception occurred (printed below), dropping item!")
                traceback.print_exception(*sys.exc_info())
                continue
            else:
                # print(mapped_item.to_dict())
                catalog.add_item(mapped_item)
    print('Finished downloads, saving catalog')
    catalog.save(catalog_type=catalog_type)
    print('wrote catalog_fpath = {!r}'.format(catalog_fpath))
    return catalog


def ingress_item(feature,
                 outdir,
                 aws_base_command,
                 dryrun,
                 relative=False,
                 virtual=False):
    """
    FIXME: Something is this is not concurrent-safe
    """
    import subprocess
    from urllib.parse import urlparse
    import pystac
    # Adding a reference back to the original STAC
    # item if not already present
    self_link = None
    has_original = False
    for link in feature.get('links', ()):
        if link['rel'] == 'self':
            self_link = link
        elif link['rel'] == 'original':
            has_original = True

    if not has_original and self_link is not None:
        feature.setdefault('links', []).append(
            {'rel': 'original',
             'href': self_link['href'],
             'type': 'application/json'})

    # Should only be added the first time the item is ingressed
    if 'watch:original_item_id' not in feature['properties']:
        feature['properties']['watch:original_item_id'] = feature['id']

    assets = feature.get('assets', {})

    # HTML index page for certain Landsat items, not needed here
    # so remove from assets dict
    if 'index' in assets:
        del assets['index']

    new_assets = {}
    assets_to_remove = set()
    for asset_name, asset in assets.items():
        asset_basename = os.path.basename(asset['href'])

        feature_output_dir = os.path.join(outdir, feature['id'])

        asset_outpath = os.path.join(feature_output_dir, asset_basename)

        asset_href = asset['href']

        try:
            if ('productmetadata' not in assets
               and feature['properties']['platform'] in SENTINEL_PLATFORMS
               and asset_name == 'metadata'):
                asset_outpath = os.path.join(
                    feature_output_dir, 'MTD_TL.xml')

                new_asset = download_mtd_msil1c(
                    feature['properties']['sentinel:product_id'],
                    asset_href, feature_output_dir, aws_base_command,
                    dryrun)

                if new_asset is not None:
                    new_assets['productmetadata'] = new_asset
        except KeyError:
            pass

        local_asset_href = os.path.abspath(asset_outpath)
        if relative:
            local_asset_href = os.path.relpath(asset_outpath, outdir)

        if not dryrun:
            os.makedirs(feature_output_dir, exist_ok=True)

        if os.path.isfile(asset_outpath):
            print('Asset already exists at outpath {!r}, '
                  'not redownloading'.format(asset_outpath))
            # Update feature asset href to point to local outpath
            asset['href'] = local_asset_href
        else:
            # Prefer to pull asset from S3 if available
            parsed_asset_href = urlparse(asset_href)
            if (parsed_asset_href.scheme != 's3'
               and 'alternate' in asset and 's3' in asset['alternate']):
                asset_href_for_download = asset['alternate']['s3']['href']
            else:
                asset_href_for_download = asset_href

            # Need to reparse the href if it switched from http to s3
            parsed_asset_href = urlparse(asset_href_for_download)

            if virtual:
                if parsed_asset_href.scheme == 's3':
                    virtual_href = '/vsis3/{}{}'.format(
                        parsed_asset_href.netloc,
                        parsed_asset_href.path)
                    # print(f'virtual_href={virtual_href}')
                    asset['href'] = virtual_href
                elif parsed_asset_href.scheme in {'http', 'https'}:
                    virtual_href = '/vsicurl/{}://{}{}'.format(
                        parsed_asset_href.scheme,
                        parsed_asset_href.netloc,
                        parsed_asset_href.path)
                    # print(f'virtual_href={virtual_href}')
                    asset['href'] = virtual_href
                else:
                    print("* Unsupported URI scheme '{}' for virtual ingress; "
                          "not updating href: {}".format(
                              parsed_asset_href.scheme, asset_href))
            else:
                try:
                    success = download_file(asset_href_for_download,
                                            asset_outpath,
                                            aws_base_command,
                                            dryrun)
                except subprocess.CalledProcessError:
                    print("* Error * Couldn't download asset from href: '{}', "
                          "removing asset from item!".format(
                              asset_href_for_download))
                    assets_to_remove.add(asset_name)
                    continue
                else:
                    if success:
                        asset['href'] = local_asset_href
                    else:
                        print('Warning unrecognized scheme for asset href: '
                              '{!r}, skipping!'.format(
                                  asset_href_for_download))
                        continue

    for asset_name in assets_to_remove:
        del assets[asset_name]

    for new_asset_name, new_asset in new_assets.items():
        assets[new_asset_name] = new_asset

    item = pystac.Item.from_dict(feature)
    item.set_collection(None)  # Clear the collection if present

    item_href = os.path.join(outdir, feature['id'], feature['id'] + '.json')
    # Transform to absolute
    item_href = os.path.abspath(item_href)
    if relative:
        # Transform to relative if requested
        item_href = os.path.relpath(item_href, outdir)

    item.set_self_href(item_href)
    # print('item = {}'.format(ub.urepr(item.to_dict(), nl=2)))
    return item


def read_input_stac_items(path):
    """
    Read the stac input format from a file on disk.

    This also handles jsonl files as well as a a fallback for whitespace
    separated data.
    """
    import json
    try:
        with open(path, 'r') as f:
            input_json = json.load(f)
        items = input_json['stac'].get('features', [])
    # Excepting KeyError here in case of a single line STAC item input
    except (json.decoder.JSONDecodeError, KeyError):
        try:
            # Support for simple newline separated STAC items
            with open(path, 'r') as f:
                items = [json.loads(line) for line in f]
        except json.decoder.JSONDecodeError:
            # Support for whitespace separated data
            with open(path, 'r') as f:
                text = f.read()
            items = []
            stack = [line for line in text.split('\n')[::-1] if line]
            while stack:
                line = stack.pop()
                try:
                    item = json.loads(line)
                except json.decoder.JSONDecodeError as e:
                    # Hack for the case where a new line is missing
                    if line[e.pos] == '{':
                        stack.append(line[e.pos:].strip())
                        stack.append(line[:e.pos])
                    else:
                        raise
                else:
                    items.append(item)
    return items


def load_input_stac_items(input_path, aws_base_command):
    """
    Load the stac input format from a file on disk or AWS
    """
    import subprocess
    import tempfile
    if input_path.startswith('s3'):
        with tempfile.NamedTemporaryFile() as temporary_file:
            subprocess.run(
                [*aws_base_command, input_path, temporary_file.name],
                check=True)

            input_stac_items = read_input_stac_items(temporary_file.name)
    else:
        input_stac_items = read_input_stac_items(input_path)

    return input_stac_items


def download_file(href, outpath, aws_base_command, dryrun):
    import subprocess
    from urllib.parse import urlparse
    # TODO: better handling of possible download failure?
    scheme, *_ = urlparse(href)
    verbose = 0
    if scheme == 's3':
        command = [*aws_base_command, href, outpath]
        if verbose > 1:
            print('Running: {}'.format(' '.join(command)))
        # TODO: Manually check return code / output
        subprocess.run(command, check=True)
    elif scheme in {'https', 'http'}:
        if verbose > 1:
            print('Downloading: {!r} to {!r}'.format(href, outpath))
        if not dryrun:
            download_http_file(href, outpath)
    else:
        return False

    return True


def download_http_file(url, outpath):
    import requests
    response = requests.get(url)
    with open(outpath, 'wb') as outf:
        for chunk in response.iter_content(chunk_size=128):
            outf.write(chunk)


def download_mtd_msil1c(product_id,
                        metadata_href,
                        outdir,
                        aws_base_command,
                        dryrun):
    from datetime import datetime as datetime_cls
    import subprocess
    from urllib.parse import urlparse
    # The metadata of the product, which tile is part of, are available in
    # parallel folder (productInfo.json contains the name of the product).
    # This can be found in products/[year]/[month]/[day]/[product name].
    # (https://roda.sentinel-hub.com/sentinel-s2-l1c/readme.html)
    try:
        dt = datetime_cls.strptime(product_id.split('_')[2], '%Y%m%dT%H%M%S')
    except ValueError:
        # Support for older format product ID format, e.g.:
        # "S2A_OPER_PRD_MSIL1C_PDMC_20160413T135705_R065_V20160412T102058_20160412T102058"
        dt = datetime_cls.strptime(product_id.split('_')[7][1:], '%Y%m%dT%H%M%S')

    scheme, netloc, path, *_ = urlparse(metadata_href)
    index = path.find('tiles')
    path = (path[:index] +
            f'products/{dt.year}/{dt.month}/{dt.day}/{product_id}/metadata.xml')
    mtd_msil1c_href = f'{scheme}://{netloc}{path}'
    mtd_msil1c_outpath = os.path.join(outdir, 'MTD_MSIL1C.xml')

    try:
        success = download_file(
            mtd_msil1c_href, mtd_msil1c_outpath, aws_base_command, dryrun)
    except subprocess.CalledProcessError:
        print('* Warning * Failed to download MTD_MSIL1C.xml')
        return None

    if success:
        return {
            'href': mtd_msil1c_outpath,
            'type': 'application/xml',
            'title': 'Product XML metadata',
            'roles': ['metadata']
        }
    else:
        print('Warning unrecognized scheme for asset href: {!r}, '
              'skipping!'.format(mtd_msil1c_href))
        return None


if __name__ == '__main__':
    main()
