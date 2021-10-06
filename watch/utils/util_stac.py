from concurrent.futures import as_completed
from typing import cast

import ubelt
import pystac


def parallel_map_items(catalog,
                       mapper_func,
                       max_workers=4,
                       mode='process',
                       extra_args=[],
                       extra_kwargs={}):
    """
    Functions similarly to `pystac.Catalog.map_items` function but
    in parallel, and allows the mapper function to return None
    indicating that the mapped STAC item should be dropped from the
    output catalog
    """
    out_catalog = catalog.full_copy()

    executor = ubelt.Executor(mode=mode, max_workers=max_workers)

    input_stac_items = []
    for item_link in catalog.get_item_links():
        item_link.resolve_stac_object(root=catalog.get_root())

        input_stac_items.append(cast(pystac.Item, item_link.target))

    jobs = [executor.submit(mapper_func, item, *extra_args, **extra_kwargs)
            for item in input_stac_items]

    output_item_links = []
    for mapped in (job.result() for job in as_completed(jobs)):
        # Allowing mappers to act as filters as well by returning None
        if mapped is None:
            continue
        elif isinstance(mapped, pystac.Item):
            output_item_links.append(pystac.Link(
                'item', mapped, pystac.MediaType.JSON))
        else:
            for i in mapped:
                output_item_links.append(pystac.Link(
                    'item', mapped, pystac.MediaType.JSON))

    out_catalog.clear_items()
    out_catalog.add_links(output_item_links)

    return out_catalog
