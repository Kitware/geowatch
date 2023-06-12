import tempfile
import subprocess
import json
import os
from urllib.parse import urlparse

import pystac

from watch.cli.baseline_framework_ingress import ingress_item
from watch.cli.baseline_framework_egress import egress_item


class CacheItemOutputS3Wrapper:
    def __init__(self, item_map, outbucket, aws_profile=None):
        self.item_map = item_map
        self.outbucket = outbucket

        if aws_profile is not None:
            self.aws_base_command = [
                'aws', 's3', '--profile', aws_profile, 'cp', '--no-progress']
        else:
            self.aws_base_command = ['aws', 's3', 'cp', '--no-progress']

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
    if aws_profile is not None:
        aws_base_command =\
            ['aws', 's3', '--profile', aws_profile, 'cp']
    else:
        aws_base_command = ['aws', 's3', 'cp']

    scheme, *_ = urlparse(input_region_path)
    if scheme == 's3':
        with tempfile.NamedTemporaryFile() as temporary_file:
            command = [*aws_base_command,
                       input_region_path,
                       temporary_file.name]

            print("Running: {}".format(' '.join(command)))
            # TODO: Manually check return code / output
            subprocess.run(command, check=True)

            with open(temporary_file.name) as f:
                out_region_data = json.load(f)
    elif scheme == '':
        with open(input_region_path) as f:
            out_region_data = json.load(f)
    else:
        raise NotImplementedError("Don't know how to pull down region file "
                                  "with URI scheme: '{}'".format(scheme))

    from watch.geoannots import geomodels

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

    References:
        https://docs.aws.amazon.com/cli/latest/reference/s3/

    Example:
        >>> from watch.utils.util_framework import *  # NOQA
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

    def run(self, verbose=3, check=True, shell=False):
        """
        Execute the S3 command

        Returns:
            Dict : ubelt cmd info dict
        """
        import ubelt as ub
        final_command = self.finalize()
        run_info = ub.cmd(final_command, verbose=3, check=check, shell=shell)
        return run_info
