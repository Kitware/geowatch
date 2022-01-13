import argparse
import sys
import json
import os
import tempfile
import subprocess
from urllib.parse import urlparse
from datetime import datetime
from concurrent.futures import as_completed

import ubelt
import requests
import pystac


SENTINEL_PLATFORMS = {'sentinel-2b', 'sentinel-2a'}


def main():
    parser = argparse.ArgumentParser(
        description='Ingress data from T&E baseline framework input file')

    parser.add_argument('input_path',
                        type=str,
                        help='Path to input T&E Baseline Framework JSON')
    parser.add_argument('-o', '--outdir',
                        type=str,
                        required=True,
                        help='Output directory for ingressed assets an output '
                             'STAC Catalog')
    parser.add_argument('--aws_profile',
                        required=False,
                        type=str,
                        help='AWS Profile to use for AWS S3 CLI commands')
    parser.add_argument('-d', '--dryrun',
                        action='store_true',
                        default=False,
                        help='Run AWS CLI commands with --dryrun flag')
    parser.add_argument('-s', '--show-progress',
                        action='store_true',
                        default=False,
                        help='Show progress for AWS CLI commands')
    parser.add_argument('-r', '--requester_pays',
                        action='store_true',
                        default=False,
                        help='Run AWS CLI commands with '
                             '`--requestor_payer requester` flag')
    parser.add_argument('-j', '--jobs',
                        type=int,
                        default=1,
                        required=False,
                        help='Number of jobs to run in parallel')

    parser.add_argument('--relative', default=False,
                        action='store_true', help='if true use relative paths')

    ns = parser.parse_args()
    ingress_kwargs = vars(ns)
    baseline_framework_ingress(**ingress_kwargs)
    return 0


def _item_map(feature, outdir, aws_base_command, dryrun, relative=False):
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

        local_asset_href = os.path.abspath(asset_outpath)
        if relative:
            local_asset_href = os.path.relpath(asset_outpath, outdir)

        asset_href = asset['href']

        try:
            if('productmetadata' not in assets
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

        if not dryrun:
            os.makedirs(feature_output_dir, exist_ok=True)

        if os.path.isfile(asset_outpath):
            print('Asset already exists at outpath {!r}, '
                  'not redownloading'.format(asset_outpath))
            # Update feature asset href to point to local outpath
            asset['href'] = local_asset_href
        else:
            # Prefer to pull asset from S3 if available
            if(urlparse(asset_href).scheme != 's3'
               and 'alternate' in asset and 's3' in asset['alternate']):
                asset_href_for_download = asset['alternate']['s3']['href']
            else:
                asset_href_for_download = asset_href

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
                    print('Warning unrecognized scheme for asset href: {!r}, '
                          'skipping!'.format(asset_href_for_download))
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
    return item


def baseline_framework_ingress(input_path,
                               outdir,
                               aws_profile=None,
                               dryrun=False,
                               show_progress=False,
                               requester_pays=False,
                               relative=False,
                               jobs=1):
    os.makedirs(outdir, exist_ok=True)

    if relative:
        catalog_type = pystac.CatalogType.RELATIVE_PUBLISHED
    else:
        catalog_type = pystac.CatalogType.ABSOLUTE_PUBLISHED

    catalog_outpath = os.path.join(outdir, 'catalog.json')
    catalog = pystac.Catalog('Baseline Framework ingress catalog',
                             'STAC catalog of SMART search results',
                             href=catalog_outpath, catalog_type=catalog_type)

    catalog.set_root(catalog)

    if relative:
        catalog.make_all_asset_hrefs_relative()

    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    if not show_progress:
        aws_base_command.append('--no-progress')

    if requester_pays:
        aws_base_command.extend(['--request-payer', 'requester'])

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
            subprocess.run(
                [*aws_base_command, input_path, temporary_file.name],
                check=True)

            input_stac_items = _load_input(temporary_file.name)
    else:
        input_stac_items = _load_input(input_path)

    executor = ubelt.Executor(mode='process' if jobs > 1 else 'serial',
                              max_workers=jobs)

    jobs = [executor.submit(_item_map, feature, outdir, aws_base_command,
                            dryrun, relative)
            for feature in input_stac_items]

    for job in as_completed(jobs):
        try:
            mapped_item = job.result()
        except Exception as e:
            print("Exception occurred (printed below), dropping item!")
            print(e)
            continue
        else:
            catalog.add_item(mapped_item)

    catalog.save(catalog_type=catalog_type)
    print('wrote catalog_outpath = {!r}'.format(catalog_outpath))
    return catalog


def download_file(href, outpath, aws_base_command, dryrun):
    # TODO: better handling of possible download failure?
    scheme, *_ = urlparse(href)

    if scheme == 's3':
        command = [*aws_base_command, href, outpath]
        print('Running: {}'.format(' '.join(command)))
        # TODO: Manually check return code / output
        subprocess.run(command, check=True)
    elif scheme in {'https', 'http'}:
        print('Downloading: {!r} to {!r}'.format(
            href, outpath))
        if not dryrun:
            download_http_file(href, outpath)
    else:
        return False

    return True


def download_http_file(url, outpath):
    response = requests.get(url)

    with open(outpath, 'wb') as outf:
        for chunk in response.iter_content(chunk_size=128):
            outf.write(chunk)


def download_mtd_msil1c(product_id,
                        metadata_href,
                        outdir,
                        aws_base_command,
                        dryrun):
    # The metadata of the product, which tile is part of, are available in
    # parallel folder (productInfo.json contains the name of the product).
    # This can be found in products/[year]/[month]/[day]/[product name].
    # (https://roda.sentinel-hub.com/sentinel-s2-l1c/readme.html)
    try:
        dt = datetime.strptime(product_id.split('_')[2], '%Y%m%dT%H%M%S')
    except ValueError:
        # Support for older format product ID format, e.g.:
        # "S2A_OPER_PRD_MSIL1C_PDMC_20160413T135705_R065_V20160412T102058_20160412T102058"
        dt = datetime.strptime(product_id.split('_')[5], '%Y%m%dT%H%M%S')

    scheme, netloc, path, *_ = urlparse(metadata_href)
    index = path.find('tiles')
    path = path[:index] + \
        f'products/{dt.year}/{dt.month}/{dt.day}/{product_id}/metadata.xml'
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
    sys.exit(main())
