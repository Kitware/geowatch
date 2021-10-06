import argparse
import sys
import json
import os
import tempfile
import subprocess
from urllib.parse import urlparse, urlunparse
from datetime import datetime

import requests
import pystac


SENTINEL_PLATFORMS = {'sentinel-2b', 'sentinel-2a'}


def main():
    parser = argparse.ArgumentParser(
        description="Ingress data from T&E baseline framework input file")

    parser.add_argument('input_path',
                        type=str,
                        help="Path to input T&E Baseline Framework JSON")
    parser.add_argument("-o", "--outdir",
                        type=str,
                        required=True,
                        help="Output directory for ingressed assets an output "
                             "STAC Catalog")
    parser.add_argument("--aws_profile",
                        required=False,
                        type=str,
                        help="AWS Profile to use for AWS S3 CLI commands")
    parser.add_argument("-d", "--dryrun",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with --dryrun flag")
    parser.add_argument("-r", "--requester_pays",
                        action='store_true',
                        default=False,
                        help="Run AWS CLI commands with "
                             "`--requestor_payer requester` flag")

    baseline_framework_ingress(**vars(parser.parse_args()))

    return 0


def baseline_framework_ingress(input_path,
                               outdir,
                               aws_profile=None,
                               dryrun=False,
                               requester_pays=False):
    os.makedirs(outdir, exist_ok=True)

    catalog_outpath = os.path.join(outdir, 'catalog.json')
    catalog = pystac.Catalog('Baseline Framework ingress catalog',
                             'STAC catalog of SMART search results',
                             href=catalog_outpath)
    catalog.set_root(catalog)

    input_path = input_path

    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    if dryrun:
        aws_base_command.append('--dryrun')

    if requester_pays:
        aws_base_command.extend(['--request-payer', 'requester'])

    def _load_input(path):
        try:
            with open(path) as f:
                input_json = json.load(f)
            return input_json['stac'].get('features', [])
        except json.decoder.JSONDecodeError:
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

    for feature in input_stac_items:
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

        assets = feature.get('assets', {})

        # HTML index page for certain Landsat items, not needed here
        # so remove from assets dict
        if 'index' in assets:
            del assets['index']

        for asset_name, asset in assets.items():
            asset_basename = os.path.basename(asset['href'])

            feature_output_dir = os.path.join(
                outdir, feature['id'])

            asset_outpath = os.path.join(
                feature_output_dir, asset_basename)

            asset_href = asset['href']

            try:
                if(feature['properties']['platform'] in SENTINEL_PLATFORMS
                   and asset_name == "metadata"):
                    asset_outpath = os.path.join(
                        feature_output_dir, "MTD_TL.xml")

                    new_asset = download_mtd_msil1c(
                        feature['properties']['sentinel:product_id'], 
                        asset_href, feature_output_dir, aws_base_command, 
                        dryrun)
                    
            except KeyError:
                pass

            if not dryrun:
                os.makedirs(feature_output_dir, exist_ok=True)

            if os.path.isfile(asset_outpath):
                print("Asset already exists at outpath '{}', "
                      "not redownloading".format(asset_outpath))
                # Update feature asset href to point to local outpath
                asset['href'] = asset_outpath
            else:
                success = download_file(
                    asset_href, asset_outpath, aws_base_command, dryrun)
                if success:
                    asset['href'] = asset_outpath
                else:
                    print("Warning unrecognized scheme for asset href: '{}', "
                          "skipping!".format(asset_href))
                    continue

        if new_asset:
            assets['productmetadata'] = new_asset

        item = pystac.Item.from_dict(feature)
        item.set_collection(None)  # Clear the collection if present
        item.set_self_href(
            os.path.join(outdir, feature['id'], feature['id'] + '.json'))
        catalog.add_item(item)

    catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

    return catalog


def download_file(href, outpath, aws_base_command, dryrun):
    # TODO: better handling of possible download failure?
    scheme, *_ = urlparse(href)

    if scheme == 's3':
        command = [*aws_base_command, href, outpath]

        print("Running: {}".format(' '.join(command)))
        # TODO: Manually check return code / output
        subprocess.run(command, check=True)
    elif scheme in {'https', 'http'}:
        print("Downloading: '{}' to '{}'".format(
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


def download_mtd_msil1c(product_id, metadata_href, outdir, aws_base_command, dryrun):
    # "The metadata of the product, which tile is part of, are available in 
    # parallel folder (productInfo.json contains the name of the product). 
    # This can be found in products/[year]/[month]/[day]/[product name]."
    # (https://roda.sentinel-hub.com/sentinel-s2-l1c/readme.html)
    dt = datetime.strptime(product_id.split('_')[2], '%Y%m%dT%H%M%S')
    
    scheme, netloc, path, *_ = urlparse(metadata_href)
    index = path.find('tiles')
    path = path[:index] + \
        f'products/{dt.year}/{dt.month}/{dt.day}/{product_id}/metadata.xml'
    mtd_msil1c_href = f'{scheme}://{netloc}/{path}'
    mtd_msil1c_outpath = os.path.join(outdir, 'MTD_MSIL1C.xml')

    success = download_file(
        mtd_msil1c_href, mtd_msil1c_outpath, aws_base_command, dryrun)
    if success:
        return {
            'href': mtd_msil1c_outpath,
            'type': 'application/xml',
            'title': 'Product XML metadata',
            'roles': ['metadata']
        }
    else:
        print("Warning unrecognized scheme for asset href: '{}', "
                "skipping!".format(mtd_msil1c_href))
        return {}


if __name__ == "__main__":
    sys.exit(main())
