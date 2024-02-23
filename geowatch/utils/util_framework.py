import tempfile
import subprocess
import json
import os
import pystac
from os.path import join


def egress_item(stac_item, outbucket, aws_base_command):
    if isinstance(stac_item, dict):
        stac_item_dict = stac_item
    elif isinstance(stac_item, pystac.Item):
        stac_item_dict = stac_item.to_dict()
    else:
        raise TypeError("Expecting 'stac_item' to be either a dictionary "
                        "or pystac.Item")

    stac_item_outpath = join(
        outbucket, "{}.json".format(stac_item.id))

    assets_outdir = join(outbucket, stac_item.id)

    for asset_name, asset in stac_item_dict.get('assets', {}).items():
        asset_basename = os.path.basename(asset['href'])

        asset_outpath = join(assets_outdir, asset_basename)

        command = [*aws_base_command, asset['href'], asset_outpath]

        print("Running: {}".format(' '.join(command)))
        # TODO: Manually check return code / output
        subprocess.run(command, check=True)

        # Update feature asset href to point to local outpath
        asset['href'] = asset_outpath

    with tempfile.NamedTemporaryFile() as temporary_file:
        with open(temporary_file.name, 'w') as f:
            print(json.dumps(stac_item_dict, indent=2), file=f)

        command = [*aws_base_command,
                   temporary_file.name, stac_item_outpath]

        subprocess.run(command, check=True)

    output_stac_item = pystac.Item.from_dict(stac_item_dict)
    output_stac_item.set_self_href(stac_item_outpath)
    return output_stac_item


def ingress_item(feature,
                 outdir,
                 aws_base_command,
                 dryrun,
                 relative=False,
                 virtual=False):
    """
    Originally from the baseline_framework_ingress code; could probably be
    cleaned up.

    FIXME: Something is this is not concurrent-safe
    """
    import subprocess
    from urllib.parse import urlparse
    import pystac
    SENTINEL_PLATFORMS = {'sentinel-2b', 'sentinel-2a'}
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


class CacheItemOutputS3Wrapper:
    def __init__(self, item_map, outbucket, aws_profile=None):
        self.item_map = item_map
        self.outbucket = outbucket

        aws_cp = AWS_S3_Command('cp')
        aws_cp.update(
            profile=aws_profile,
            no_progress=True,
        )
        self.aws_base_command = aws_cp.finalize()

    def __call__(self, stac_item, *args, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdirname:
            status_file_basename = '{}.done'.format(stac_item['id'])
            status_item_s3_path = os.path.join(
                self.outbucket, 'status', status_file_basename)
            status_item_local_path = os.path.join(
                tmpdirname, status_file_basename)

            try:
                subprocess.run([*self.aws_base_command,
                                status_item_s3_path,
                                status_item_local_path],
                               check=True)
            except subprocess.CalledProcessError:
                pass
            else:
                print("* Item: {} previously processed, not "
                      "re-processing".format(stac_item['id']))
                with open(status_item_local_path) as f:
                    return [json.loads(line) for line in f]

            output_stac_items = self.item_map(stac_item, *args, **kwargs)

            output_status_file = os.path.join(
                tmpdirname, '{}.output.done'.format(stac_item['id']))
            with open(output_status_file, 'w') as outf:
                if isinstance(output_stac_items, dict):
                    print(json.dumps(output_stac_items), file=outf)
                elif isinstance(output_stac_items, pystac.Item):
                    print(json.dumps(output_stac_items.to_dict()), file=outf)
                else:
                    for output_item in output_stac_items:
                        if isinstance(output_item, pystac.Item):
                            print(json.dumps(output_item.to_dict()), file=outf)
                        else:
                            print(json.dumps(output_item), file=outf)

            subprocess.run([*self.aws_base_command,
                            output_status_file,
                            status_item_s3_path], check=True)

            return output_stac_items


def _default_item_selector(stac_item):
    return True


def _default_asset_selector(asset_name, asset):
    return True


class IngressProcessEgressWrapper:
    def __init__(self,
                 item_map,
                 outbucket,
                 aws_base_command,
                 dryrun=False,
                 stac_item_selector=_default_item_selector,
                 asset_selector=_default_asset_selector,
                 skip_egress=False):
        self.item_map = item_map
        self.outbucket = outbucket
        self.aws_base_command = aws_base_command
        self.dryrun = dryrun
        self.stac_item_selector = stac_item_selector
        self.asset_selector = asset_selector
        self.skip_egress = skip_egress

    def __call__(self, stac_item, *args, **kwargs):
        # Assumes that the 'self.item_map' function accepts
        # 'stac_item' and 'working_dir' arguments. TODO: actually
        # check this via introspection
        print("* Processing item: {}".format(stac_item['id']))

        if not self.stac_item_selector(stac_item):
            print("**  STAC item {} did not satisfy selector, not "
                  "processing".format(stac_item['id']))
            return [stac_item]

        with tempfile.TemporaryDirectory() as tmpdirname:
            ingressed_item = ingress_item(
                stac_item,
                os.path.join(tmpdirname, 'ingress'),
                self.aws_base_command,
                self.dryrun,
                relative=False,
                asset_selector=self.asset_selector)

            # Stripping the 'root' link here as it usually refers to
            # the catalog which isn't ingressed when we call
            # ingress_item directly (can throw exceptions when trying
            # to convert to dict or serialize when the catalog is
            # missing)
            ingressed_item.remove_links('root')

            processed_item = self.item_map(
                ingressed_item,
                tmpdirname,
                *args, **kwargs)

            processed_items = []
            if isinstance(processed_item, dict):
                processed_items.append(pystac.Item.from_dict(processed_item))
            elif isinstance(processed_item, pystac.Item):
                processed_items.append(processed_item)
            else:
                # Assume already an iterable of pystac.Item
                processed_items = processed_item

            if self.skip_egress:
                return processed_items

            output_items = []
            for item in processed_items:
                output_items.append(egress_item(item,
                                                self.outbucket,
                                                self.aws_base_command))

            # Returning a list here
            return output_items


def download_region(input_region_path,
                    output_region_path,
                    aws_profile=None,
                    strip_nonregions=False,
                    ensure_comments=False):
    from geowatch.utils.util_fsspec import FSPath
    from geowatch.geoannots import geomodels

    # TODO: handle aws_profile
    assert aws_profile is None, 'unhandled'

    input_region_path = FSPath.coerce(input_region_path)
    with input_region_path.open('r') as file:
        out_region_data = json.load(file)

    region = geomodels.RegionModel.coerce(out_region_data)
    region.fixup()

    if strip_nonregions:
        region.strip_body_features()

    if ensure_comments:
        region.ensure_comments()

    with open(output_region_path, 'w') as f:
        print(region.dumps(indent=2), file=f)

    return output_region_path


def determine_region_id(region_fpath):
    """
    Args:
        region_fpath (str | PathLike):
            the path to a region model geojson file

    Returns:
        str | None : the region id if we can find one
    """
    region_id = None
    with open(region_fpath, 'r') as file:
        region_data = json.load(file)
        for feature in region_data.get('features', []):
            props = feature['properties']
            if props['type'] == 'region':
                region_id = props.get('region_id', props.get('region_model_id'))
                break
    return region_id


class AWS_S3_Command:
    """
    Helper to build and execute AWS S3 bash commands

    Note:
        probably should use fsspec instead of this in most cases.

    References:
        https://docs.aws.amazon.com/cli/latest/reference/s3/

    Example:
        >>> from geowatch.utils.util_framework import *  # NOQA
        >>> self = AWS_S3_Command('ls', 's3://foo/bar')
        >>> self.update(profile='myprofile')
        >>> print(self.finalize())
        ['aws', 's3', '--profile', 'myprofile', 'ls', 's3://foo/bar']
        >>> self = AWS_S3_Command('cp', 's3://foo/bar', '/foo/bar', quiet=True, no_progress=True, color='auto')
        >>> print(self.finalize())
        ['aws', 's3', '--quiet', '--no-progress', '--color', 'auto', 'cp', 's3://foo/bar', '/foo/bar']

    Example:
        >>> # Reuse the same command object with different positional args
        >>> aws_cmd = AWS_S3_Command('cp')
        >>> aws_cmd.update(
        >>>     profile='myprof',
        >>>     only_show_errors=True
        >>> )
        >>> aws_cmd.args = ['s3://data1', '/local/data1']
        >>> print(aws_cmd.finalize())
        ['aws', 's3', '--only-show-errors', '--profile', 'myprof', 'cp', 's3://data1', '/local/data1']
        >>> # Set the `args` attribute to get a new command while keeping
        >>> # existing options.
        >>> aws_cmd.update(recursive=True)
        >>> aws_cmd.args = ['s3://data2', '/local/data2']
        >>> print(aws_cmd.finalize())
        ['aws', 's3', '--only-show-errors', '--recursive', '--profile', 'myprof', 'cp', 's3://data2', '/local/data2']

    Example:
        >>> # There is no need to specify the entire command. If you want
        >>> # to simply build a command prefix, then that works too.
        >>> aws_cmd = AWS_S3_Command('cp', profile='myprof', aws_storage_class='foobar')
        >>> print(aws_cmd.finalize())
        ['aws', 's3', '--profile', 'myprof', '--aws-storage-class', 'foobar', 'cp']
    """

    # Register known options for known commands
    # TODO: multi values

    cmd_known_flags = {}
    cmd_known_keyvals = {}

    cmd_known_flags['ls'] = [
        'recursive',
        'human-readable',
        'summarize',
        'debug',
        'no-verify-ssl',
        'no-paginate',
        'no-sign-request',
    ]

    cmd_known_keyvals['ls'] = [
        'endpoint-url',
        'page-size',
        'request-payer',
        'output',
        'query',
        'profile',
        'region',
        'version',
        'color',
        'ca-bundle',
        'cli-read-timeout',
        'cli-connect-timeout',
    ]

    cmd_known_flags['sync'] = [
        'dryrun',
        'quiet',
        'follow-symlinks',
        'no-follow-symlinks',
        'no-guess-mime-type',
        'only-show-errors',
        'no-progress',
        'ignore-glacier-warnings',
        'force-glacier-transfer',
        'size-only',
        'exact-timestamps',
        'delete',
        'debug',
        'no-verify-ssl',
        'no-paginate',
        'no-sign-request',
    ]

    cmd_known_keyvals['sync'] = [
        'include',
        'exclude',
        'acl',
        'sse',
        'sse-c',
        'sse-c-key',
        'sse-kms-key-id',
        'sse-c-copy-source',
        'sse-c-copy-source-key',
        'storage-class',
        'grants',
        'website-redirect',
        'content-type',
        'cache-control',
        'content-disposition',
        'content-encoding',
        'content-language',
        'expires',
        'source-region',
        'page-size',
        'request-payer',
        'metadata',
        'metadata-directive',
        'endpoint-url',
        'output',
        'query',
        'profile',
        'region',
        'version',
        'color',
        'ca-bundle',
        'cli-read-timeout',
        'cli-connect-timeout',
    ]

    cmd_known_flags['cp'] = [
        'dryrun',
        'quiet',
        'follow-symlinks',
        'no-follow-symlinks',
        'no-guess-mime-type',
        'only-show-errors',
        'no-progress',
        'ignore-glacier-warnings',
        'force-glacier-transfer',
        'recursive',
        'debug',
        'no-verify-ssl',
        'no-paginate',
    ]

    cmd_known_keyvals['cp'] = [
        'include',
        'exclude',
        'acl',
        'sse',
        'sse-c',
        'sse-c-key',
        'sse-kms-key-id',
        'sse-c-copy-source',
        'sse-c-copy-source-key',
        'storage-class',
        'grants',
        'website-redirect',
        'content-type',
        'cache-control',
        'content-disposition',
        'content-encoding',
        'content-language',
        'expires',
        'source-region',
        'page-size',
        'request-payer',
        'metadata',
        'metadata-directive',
        'expected-size',
        'endpoint-url',
        'output',
        'query',
        'profile',
        'region',
        'version',
        'color',
        'no-sign-request',
        'ca-bundle',
        'cli-read-timeout',
        'cli-connect-timeout',
    ]

    def __init__(self, command, *args, **options):
        """
        Args:
            command (str):
                can be: cp, ls, mv, rm, sync

            *args: positional arguments

            **options: key value options (e.g. profile)
        """
        self.command = command
        self.args = args

        self._known_flags = self.cmd_known_flags.get(self.command, [])
        self._known_keyvals = self.cmd_known_keyvals.get(self.command, [])
        self._known_flags = self._known_flags + [k.replace('-', '_') for k in self._known_flags]
        self._known_keyvals = self._known_keyvals + [k.replace('-', '_') for k in self._known_keyvals]

        # Key / value CLI arguments
        self._keyval_options = {}
        self._flag_options = {}

        self.update(options)

    def update(self, arg=None, /, **options):
        """
        Update key / value options.

        This function is aware of what options need to be flags versus key/values

        So quiet=True will result in `--quiet`, quiet=False will have include no option.
        Likewise profile=foo will result in `--profile foo` and profile=None will include no option.
        """
        if arg is not None:
            import ubelt as ub
            options = ub.udict(arg) | options

        for k, v in options.items():
            if k in self._known_flags:
                self._flag_options[k] = v
            elif k in self._known_keyvals:
                self._keyval_options[k] = v
            else:
                # Fallback to key/value
                self._keyval_options[k] = v

    def finalize(self):
        """
        Returns:
            List[str]: commands suitable for passing to POpen
        """
        parts = ['aws', 's3']
        for k, v in self._flag_options.items():
            if v:
                parts.extend(['--' + k.replace('_', '-')])
        for k, v in self._keyval_options.items():
            if v is not None:
                parts.extend(['--' + k.replace('_', '-'), v])
        parts.append(self.command)
        parts.extend(self.args)
        return parts

    def run(self, check=True, shell=False, capture=False, verbose=3):
        """
        Execute the S3 command

        Returns:
            Dict : ubelt cmd info dict
        """
        import ubelt as ub
        final_command = self.finalize()
        print('final_command = {}'.format(ub.urepr(final_command, nl=1)))
        run_info = ub.cmd(final_command, verbose=verbose, shell=shell,
                          capture=capture)
        if check:
            run_info.check_returncode()
        return run_info


def ta2_collate_output(aws_base_command, local_region_dir, local_sites_dir,
                       destination_s3_bucket, performer_suffix='KIT'):
    """
    I think this is for putting the final system regions / sites into the place
    that T&E wants them.
    """
    from glob import glob
    from geowatch.utils import util_fsspec
    assert aws_base_command is None, 'unused'

    def _get_suffixed_basename(local_path):
        base, ext = os.path.splitext(os.path.basename(local_path))
        if performer_suffix is not None and performer_suffix != '':
            return "{}_{}{}".format(base, performer_suffix, ext)
        else:
            return "{}{}".format(base, ext)

    for region in glob(join(local_region_dir, '*.geojson')):

        region_s3_outpath = '/'.join((destination_s3_bucket,
                                      'region_models',
                                      _get_suffixed_basename(region)))
        region = util_fsspec.FSPath.coerce(region)
        region_s3_outpath = util_fsspec.FSPath.coerce(region_s3_outpath)
        region.copy(region_s3_outpath)

    for site in glob(join(local_sites_dir, '*.geojson')):
        site_s3_outpath = '/'.join((destination_s3_bucket,
                                    'site_models',
                                    _get_suffixed_basename(site)))
        site = util_fsspec.FSPath.coerce(site)
        site_s3_outpath = util_fsspec.FSPath.coerce(site_s3_outpath)
        site.copy(site_s3_outpath)


def fixup_and_validate_site_and_region_models(region_dpath, site_dpath):
    """
    Read, fix, and validate all site and region models.
    """
    # Validate and fix all outputs
    from geowatch.geoannots import geomodels
    from geowatch.utils import util_gis
    region_infos = list(util_gis.coerce_geojson_datas(region_dpath, format='json'))
    site_infos = list(util_gis.coerce_geojson_datas(site_dpath, format='json'))
    for region_info in region_infos:
        fpath = region_info['fpath']
        region = geomodels.RegionModel(**region_info['data'])
        region.fixup()
        fpath.write_text(region.dumps(indent='    '))
        region.validate()
    for site_info in site_infos:
        fpath = site_info['fpath']
        site = geomodels.SiteModel(**site_info['data'])
        site.fixup()
        fpath.write_text(site.dumps(indent='    '))
        site.validate()


class NodeStateDebugger:
    """
    Prints information about the current node that is helpful for debugging.

    Use in the smartflow CLI nodes.

    Maintains some internal state to keep things organized.

    Example:
        >>> from geowatch.utils.util_framework import *  # NOQA
        >>> import ubelt as ub
        >>> watch_appdir_dpath = ub.Path.appdir('geowatch')
        >>> self = NodeStateDebugger()
        >>> self.print_environment()
        >>> self.print_current_state(watch_appdir_dpath)
        >>> self.print_current_state(watch_appdir_dpath)
    """

    def __init__(self):
        self.current_iteration = 0

    def print_environment(self):
        # Print info about what version of the code we are running on
        import ubelt as ub
        import os
        import geowatch
        print(' --- <NODE_ENV> --- ')
        print(' * Print current version of the code & environment')
        ub.cmd('git log -n 1', verbose=3, cwd=ub.Path(geowatch.__file__).parent)
        print('geowatch.__version__ = {}'.format(ub.urepr(geowatch.__version__, nl=1)))
        print('geowatch.__file__ = {}'.format(ub.urepr(geowatch.__file__, nl=1)))
        print('os.environ = {}'.format(ub.urepr(dict(os.environ), nl=1)))

        # Check to make sure our times are in sync with amazon servers
        if 0:
            ub.cmd('date -u', verbose=3)
            ub.cmd('curl http://s3.amazonaws.com -v', verbose=3)
        print(' --- </NODE_ENV> --- ')

    def print_current_state(self, dpath):
        import ubelt as ub
        print(f' --- <NODE_STATE iter={self.current_iteration}> --- ')
        print(f'* Printing current directory contents ({self.current_iteration})')
        dpath = ub.Path(dpath).resolve()
        # cwd_paths = sorted([p.resolve() for p in dpath.glob('*')])
        # print('cwd_paths = {}'.format(ub.urepr(cwd_paths, nl=1)))
        if dpath.exists():
            ub.cmd('ls -al', verbose=3, cwd=dpath)
        else:
            print(f'dpath={dpath} does not exist')

        print(f' * Print some disk and machine statistics ({self.current_iteration})')
        ub.cmd('df -h', verbose=3)

        from geowatch.utils import util_hardware
        mem_info = util_hardware.get_mem_info()
        print('mem_info = {}'.format(ub.urepr(mem_info, nl=1, align=':')))

        print(f' --- </NODE_STATE iter={self.current_iteration}> --- ')
        self.current_iteration += 1


class PrintLogger:
    """
    Ducktype a logger
    """
    def info(self, msg, *args, **kwargs):
        print(msg % args)
    def debug(self, msg, *args, **kwargs):
        print(msg % args)
    def error(self, msg, *args, **kwargs):
        print(msg % args)
    def warning(self, msg, *args, **kwargs):
        print(msg % args)
    def critical(self, msg, *args, **kwargs):
        print(msg % args)


def _devcheck_retry():
    class Dummy:
        def __init__(self):
            self.count = 0

        def func_to_run(self):
            self.count += 1
            if self.count < 3:
                raise Exception('exception')
    self = Dummy()
    from retry.api import retry_call

    logger = PrintLogger()
    retry_call(self.func_to_run, fargs=[],
               fkwargs=dict(), tries=4,
               exceptions=(Exception,), delay=3, logger=logger)


def _test_s3_hack():
    """
    An issue that can occur in will manifest as:

    [2023-10-27, 23:28:51 UTC] {pod_manager.py:342} INFO - botocore.exceptions.ClientError: An error occurred (RequestTimeTooSkewed) when calling the PutObject operation: The difference between the request time and the current time is too large.


    botocore.exceptions.ClientError: An error occurred (RequestTimeTooSkewed) when calling the PutObject operation: The difference between the request time and the current time is too large.
    File "/root/code/watch/geowatch/cli/smartflow_egress.py", line 174, in smartflow_egress
    local_path.copy(asset_s3_outpath)
    File "/root/.pyenv/versions/3.11.2/lib/python3.11/site-packages/s3fs/core.py", line 140, in _error_wrapper
    raise err
    PermissionError: The difference between the request time and the current time is too large.
    """
    from geowatch.utils import util_fsspec
    util_fsspec.S3Path._new_fs(profile='iarpa')
    s3_dpath = util_fsspec.S3Path.coerce('s3://smartflow-023300502152-us-west-2/smartflow/env/kw-v3-0-0/work/preeval17_batch_v120/batch/kit/CN_C000/2021-08-31/split/mono/products/dummy-test')
    s3_dpath.parent.ls()
    dst_dpath = s3_dpath

    dpath = util_fsspec.LocalPath.appdir('geowatch/fsspec/test-s3-hack/').ensuredir()
    # dst_dpath = (dpath / 'dst')

    src_dpath = (dpath / 'src').ensuredir()
    for i in range(100):
        (src_dpath / f'file_{i:03d}.txt').write_text('hello world' * 100)

    from fsspec.callbacks import TqdmCallback
    callback = TqdmCallback(tqdm_kwargs={"desc": "Your tqdm description"})
    src_dpath.copy(dst_dpath, callback=callback)
    from geowatch.utils import util_framework

    aws_cmd = util_framework.AWS_S3_Command(
        'sync', src_dpath, dst_dpath,
        **dst_dpath.fs.storage_options)
    print(aws_cmd.finalize())
    aws_cmd.run()
