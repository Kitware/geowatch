import argparse
import os
import subprocess
import sys

import pystac
import ubelt as ub


def main():
    parser = argparse.ArgumentParser(
        description="Align all STAC item data assets to the same CRS")

    parser.add_argument('stac_catalog',
                        type=str,
                        help="Path to input STAC catalog")
    parser.add_argument("--s3_bucket",
                        type=str,
                        required=True,
                        help="S3 path prefix where the STAC catalog and data "
                             "should be uploaded")
    parser.add_argument("--dry_run",
                        action='store_true',
                        help="Don't actually upload, just echo AWS commands")
    parser.add_argument("-o", "--outdir",
                        type=str,
                        required=False,
                        help="Output directory for STAC catalog with s3 "
                             "egress hrefs")

    stac_egress(**vars(parser.parse_args()))

    return 0


def stac_egress(stac_catalog,
                s3_bucket,
                dry_run=False,
                outdir=None):
    if isinstance(stac_catalog, str):
        catalog = pystac.read_file(href=stac_catalog).full_copy()
    else:
        catalog = stac_catalog.full_copy()

    # Ensures that root and parent paths are pointing to the uploaded
    # path
    catalog.normalize_hrefs(s3_bucket)

    # Working directory
    if outdir is None:
        working_dir = ub.ensure_app_cache_dir(
            'watch/tools/stac_egress_working')
        ub.delete(working_dir)  # remove the dir and contents if it exists
        ub.ensuredir(working_dir)  # create the empty directory.
    else:
        working_dir = outdir
        os.makedirs(working_dir, exist_ok=True)

    def _item_map(stac_item):
        item_outdir = os.path.join(s3_bucket, stac_item.id)
        item_outpath = os.path.join(
            item_outdir, "{}.json".format(stac_item.id))

        for asset_name, asset in stac_item.assets.items():
            asset_outpath = os.path.join(
                item_outdir, os.path.basename(asset.href))

            command = ['aws', 's3', '--profile', 'iarpa', 'cp']
            if dry_run:
                command.append('--dryrun')

            command.extend([asset.href, asset_outpath])

            # TODO: Manually check return code / output
            subprocess.run(command, check=True)

            asset.href = asset_outpath

        # 'previous' links typically refer to transient items on the
        # worker node; probably doesn't make sense to persist them by
        # uploading to S3
        stac_item.remove_links('previous')

        stac_item.set_self_href(item_outpath)
        stac_item_tmp_path = os.path.join(
            working_dir,
            stac_item.id,
            "{}.json".format(stac_item.id))

        # Write temporarily to working directory for upload
        pystac.write_file(
            stac_item,
            include_self_link=True,
            dest_href=stac_item_tmp_path)

        command = ['aws', 's3', '--profile', 'iarpa', 'cp']
        if dry_run:
            command.append('--dryrun')

        command.extend([stac_item_tmp_path, item_outpath])

        # TODO: Manually check return code / output
        subprocess.run(command, check=True)

        return stac_item

    output_catalog = catalog.map_items(_item_map)

    output_catalog_path = os.path.join(s3_bucket, 'catalog.json')
    output_catalog_tmp_path = os.path.join(working_dir, 'catalog.json')

    # Write temporarily to working directory for upload
    pystac.write_file(
        output_catalog,
        include_self_link=True,
        dest_href=output_catalog_tmp_path)

    command = ['aws', 's3', '--profile', 'iarpa', 'cp']
    if dry_run:
        command.append('--dryrun')

    command.extend([output_catalog_tmp_path, output_catalog_path])

    # TODO: Manually check return code / output
    subprocess.run(command, check=True)

    if outdir is None:
        # remove the temporary working dir and contents if it exists
        ub.delete(working_dir)

    return output_catalog


if __name__ == "__main__":
    sys.exit(main())
