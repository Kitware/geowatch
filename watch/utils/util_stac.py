from concurrent.futures import as_completed
from typing import cast

import ubelt as ub
import pystac


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
