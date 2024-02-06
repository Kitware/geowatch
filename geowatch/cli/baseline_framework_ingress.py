#!/usr/bin/env python3
import sys
import os
import scriptconfig as scfg
import ubelt as ub


class BaselineFrameworkIngressConfig(scfg.DataConfig):
    """
    Ingress data from T&E baseline framework input file. The output will be stored as a json catalog
    """
    input_path = scfg.Value(None, type=str, position=1, help=ub.paragraph(
            '''
            Path to input T&E Baseline Framework JSON
            '''))
    outdir = scfg.Value(None, type=str, short_alias=['o'], help=ub.paragraph(
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
        if self.catalog_fpath is not None and self.outdir is None:
            self.outdir = os.path.dirname(os.path.abspath(self.catalog_fpath))


def main():
    config = BaselineFrameworkIngressConfig.cli(strict=True)
    import rich
    rich.print(ub.urepr(config))

    if config.outdir is None:
        raise ValueError('outdir is required')

    if config.input_path is None:
        raise ValueError('input_path is required')

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

    from kwutil import util_parallel
    from kwutil import util_progress
    import rich
    import pystac
    import traceback
    from geowatch.utils import util_fsspec
    from geowatch.utils.util_framework import ingress_item

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

    if aws_profile is not None or requester_pays:
        # This should be sufficient, but it is not tested.
        util_fsspec.S3Path._new_fs(
            profile=aws_profile, requester_pays=requester_pays)

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
    pman = util_progress.ProgressManager(backend='rich')
    with pman:
        """
        DEVELOPER NOTE:
            There is something that can cause a lockup here. To reproduce
            first ensure that the outdir is cleared, so no caching happens.
            The failure seems to happen when the mode is process. Using thread
            or serial seems fine.

            Update: the issue seems to happen if you use the pool.__enter__
            method before pman. Using it after seems ok with progiter, but not
            rich. Removing the __enter__ does not help the rich case, and now
            switching back to progiter, the lockup is happening again...
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


def read_input_stac_items(path):
    """
    Read the stac input format from a file on disk.

    This also handles jsonl files as well as a a fallback for whitespace
    separated data.
    """
    import json

    def _open(p, m):
        # handle fsspath objects gracefully
        if hasattr(path, 'open'):
            return path.open(m)
        else:
            return open(path, m)
    try:
        with _open(path, 'r') as f:
            input_json = json.load(f)
        items = input_json['stac'].get('features', [])
    # Excepting KeyError here in case of a single line STAC item input
    except (json.decoder.JSONDecodeError, KeyError):
        try:
            # Support for simple newline separated STAC items
            with _open(path, 'r') as f:
                items = [json.loads(line) for line in f]
        except json.decoder.JSONDecodeError:
            # Support for whitespace separated data
            with _open(path, 'r') as f:
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
    import tempfile

    if aws_base_command is None or not input_path.startswith('s3'):
        # New method
        from geowatch.utils import util_fsspec
        input_path = util_fsspec.FSPath.coerce(input_path)
        input_stac_items = read_input_stac_items(input_path)
    else:
        # Old method, remove once we can
        with tempfile.NamedTemporaryFile() as temporary_file:
            ub.cmd(
                [*aws_base_command, input_path, temporary_file.name],
                check=True, verbose=3)

            input_stac_items = read_input_stac_items(temporary_file.name)

    return input_stac_items


if __name__ == '__main__':
    main()
