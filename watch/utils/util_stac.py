from concurrent.futures import as_completed
from typing import cast
import subprocess
import tempfile
import json

import os
import functools
import ubelt as ub
import pystac
from inspect import signature


def parallel_map_items(catalog,
                       mapper_func,
                       max_workers=4,
                       mode='process',
                       drop_on_error=True,
                       extra_args=[],
                       extra_kwargs={}):
    """
    Functions similarly to `pystac.Catalog.map_items` function but
    in parallel, and allows the mapper function to return None
    indicating that the mapped STAC item should be dropped from the
    output catalog

    Args:
        catalog (pystac.Catalog): catalog to apply transform to

        mapper_func (callable): function to be applied to each STAC item.
            The first positional argument must accept a :class:`pystac.Item`,
            and may take any arbitrary additional positional or keyword
            arguments.

        max_workers (int): number of jobs

        mode (str): process, thread, or serial

        extra_args (List | Tuple):
            extra positional args passed to mapper_func

        extra_kwargs (Dict[str, object]):
            extra keyword args passed to mapper_func

    Returns:
        pystac.Catalog: modified catalog

    Example:
        >>> from watch.utils.util_stac import *  # NOQA
        >>> from watch.demo import stac_demo
        >>> catalog_fpath = stac_demo.demo()
        >>> catalog = pystac.Catalog.from_file(catalog_fpath)
        >>> def demo_mapper_func(item):
        >>>     import copy
        >>>     print('Process: item = {}'.format(ub.repr2(item, nl=1)))
        >>>     print('item.assets = {}'.format(ub.repr2(item.assets, nl=1)))
        >>>     if 'data' not in item.assets:
        >>>         print('Drop asset without data')
        >>>         return None  # drop assets without data
        >>>     # Pretend we do some image operation and write to a new path
        >>>     in_fpath = item.assets['data'].href
        >>>     out_fpath = ub.augpath(in_fpath, suffix='_demo_process')
        >>>     print('in_fpath = {!r}'.format(in_fpath))
        >>>     out_fpath = in_fpath
        >>>     new_item = copy.deepcopy(item)
        >>>     new_item.assets['data'].href = out_fpath
        >>>     return item
        >>> out_catalog = parallel_map_items(
        >>>     catalog, demo_mapper_func, mode='serial')
        >>> assert len(list(catalog.get_all_items())) == 2, 'two items in'
        >>> assert len(list(out_catalog.get_all_items())) == 1, 'one item out'
    """
    out_catalog = catalog.full_copy()

    input_stac_items = []
    for item_link in catalog.get_item_links():
        item_link.resolve_stac_object(root=catalog.get_root())
        item = cast(pystac.Item, item_link.target)
        input_stac_items.append(item)

    executor = ub.Executor(mode=mode, max_workers=max_workers)

    jobs = [executor.submit(mapper_func, item, *extra_args, **extra_kwargs)
            for item in input_stac_items]

    output_item_links = []
    for job in as_completed(jobs):
        try:
            mapped = job.result()
        except Exception as e:
            if drop_on_error:
                print("Exception occurred (printed below), dropping item!")
                print(e)
                continue
            else:
                raise e
        else:
            # Allowing mappers to act as filters as well by returning None
            if mapped is None:
                continue
            elif isinstance(mapped, pystac.Item):
                output_item_links.append(pystac.Link(
                    'item', mapped, pystac.MediaType.JSON))
            else:
                for item in mapped:
                    output_item_links.append(pystac.Link(
                        'item', item, pystac.MediaType.JSON))

    out_catalog.clear_items()
    out_catalog.add_links(output_item_links)

    return out_catalog


def maps(_item_map=None, history_entry=None):
    '''
    General-purpose wrapper for STAC _item_maps.

    An _item_map should take in a STAC item and return a STAC item or None.
    To support this, it should have an arg 'stac_item' and an arg or kwarg
    'outdir'.

    This decorator handles the following tasks:
        - add original item link
        - create an item_outdir from a base outdir, and pass it to _item_map.
        - update item's self_href
        - add an entry to 'watch:process_history'

    References:
        https://pybit.es/articles/decorator-optional-argument/
        https://docs.python.org/3/library/inspect.html#inspect.BoundArguments
    '''
    if _item_map is None:
        return functools.partial(maps, history_entry=history_entry)

    if history_entry is None:
        history_entry = _item_map.__name__

    @functools.wraps(_item_map)
    def wrapper(*args, **kwargs):

        try:
            bound_args = signature(_item_map).bind(*args, **kwargs)
            bound_args.apply_defaults()
            stac_item = bound_args.arguments['stac_item']
            outdir = bound_args.arguments['outdir']
        except (IndexError, KeyError):
            raise ValueError(
                f'_item_map {_item_map.__name__} must have arg "stac_item", '
                'arg or kwarg "outdir"')

        # This assumes we're not changing the stac_item ID in any of
        # the mapping functions
        item_outdir = os.path.join(outdir, stac_item.id)
        os.makedirs(item_outdir, exist_ok=True)
        bound_args.arguments['outdir'] = item_outdir

        # Adding a reference back to the original STAC
        # item if not already present
        if len(stac_item.get_links('original')) == 0:
            stac_item.links.append(
                pystac.Link.from_dict({
                    'rel': 'original',
                    'href': stac_item.get_self_href(),
                    'type': 'application/json'
                }))
        bound_args.arguments['stac_item'] = stac_item

        output_stac_item = _item_map(*bound_args.args, **bound_args.kwargs)

        if output_stac_item is not None:
            output_stac_item.set_self_href(
                os.path.join(item_outdir,
                             "{}.json".format(output_stac_item.id)))

            # Roughly keeping track of what WATCH processes have been
            # run on this particular item
            output_stac_item.properties.setdefault('watch:process_history',
                                                   []).append(history_entry)

        return output_stac_item

    return wrapper


class CacheItemOutputS3:
    def __init__(self, item_map, outbucket, aws_profile=None):
        self.item_map = item_map
        self.outbucket = outbucket

        if aws_profile is not None:
            self.aws_base_command =\
              ['aws', 's3', '--profile', aws_profile, 'cp', '--no-progress']
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
                for output_item in output_stac_items:
                    if isinstance(output_item, pystac.Item):
                        print(json.dumps(output_item.to_dict()), file=outf)
                    else:
                        print(json.dumps(output_item.to_dict()), file=outf)

            subprocess.run([*self.aws_base_command,
                            output_status_file,
                            status_item_s3_path], check=True)

            return output_stac_items


def associate_msi_pan(stac_catalog):
    '''
    Match up WorldView multispectral and panchromatic items.

    Returns a dict {msi_item.id: pan_item}, where pan_item can be
    nonunique.
    '''

    # more efficient way to do this if collections are preserved during
    # intermediate steps:
    # search = catalog.search(collections=['worldview-nitf'])
    # items = list(search.get_items()
    items_dct = {
        i.id: i
        for i in stac_catalog.get_all_items()
        if i.properties.get('constellation') == 'worldview'
    }

    if 0:

        # more robust way of matching PAN->MSI items one-to-many

        import pandas as pd
        from parse import parse

        df = pd.DataFrame.from_records(
            [item.properties for item in items_dct.values()])
        df['id'] = list(items_dct.keys())

        def _part(_id):
            result = parse('{}_P{part:3d}', _id)
            if result is not None:
                return result['part']
            else:
                return -1

        df['part'] = list(map(_part, df['id']))

        def _vnir_source(source):
            if source in {
                    'DigitalGlobe Acquired Image',
                    'DigitalGlobe Acquired Imagery'
            }:
                return -1
            for s in source.split(', '):
                try:
                    p = parse('{instr:l}: {rest}', s)
                    assert p['instr'] in {'SWIR', 'VNIR', 'CAVIS'}, source
                    if p['instr'] == 'VNIR':
                        return p['rest']
                except KeyError:
                    print(s, p)
            return -1

        df['vnir_source'] = list(map(_vnir_source, df['nitf:source']))

        df['geometry'] = [items_dct[i].geometry for i in df['id']]

        raise NotImplementedError

    else:

        # hacky way of matching up items by ID. Only works for pairs of 1 PAN
        # and 1 MSI, and only some sensors.
        # this matches up 40152/52563 items in the catalog.
        #
        # There are 29000 PAN items, so it matches about 2/3 of PAN.
        # The rest, besides different naming schemes, may be accounted for
        # by the fact that PAN and MSI taken during the same collect
        # can have different spatial tiling.
        mp_dct = {}
        for _id in items_dct:
            code = _id[14]
            if code != 'P':
                pid = _id[:14] + 'P' + _id[15:]
                if pid in items_dct:
                    mp_dct[_id] = items_dct[pid]
        return mp_dct
