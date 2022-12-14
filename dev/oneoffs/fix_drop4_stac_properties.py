def main():
    """
    Enrich previous datasets with STAC metadata that should now be populated by
    default.
    """
    import watch
    import kwcoco
    import os
    dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    bundle_dpath = dvc_dpath / 'Drop4-BAS'

    src = bundle_dpath / 'data_train.kwcoco.json'
    # src = bundle_dpath / 'data_vali.kwcoco.json'
    src = bundle_dpath / 'data.kwcoco.json'
    dset = kwcoco.CocoDataset.coerce(src)

    import pystac_client
    headers = {
        'x-api-key': os.environ['SMART_STAC_API_KEY']
    }
    provider = "https://api.smart-stac.com"
    catalog = pystac_client.Client.open(provider, headers=headers)

    sensor_to_collection = {
        'L8': 'ta1-ls-acc',
        'S2': 'ta1-s2-acc',
        'WV': 'ta1-wv-acc',
    }

    from watch.utils import util_time
    import ubelt as ub

    _cache = {}

    for coco_img in ub.ProgIter(dset.images().coco_images, adjust=False, freq=1):
        sensor = coco_img.img.get('sensor_coarse', None)
        date_captured = coco_img.img.get('date_captured', None)
        date_captured = util_time.coerce_datetime(date_captured)
        date = date_captured.date()

        start = date.isoformat()
        end = date.isoformat()

        collection = sensor_to_collection[sensor]

        stac_ids = set()
        for asset in coco_img.img['auxiliary']:
            parent_fnames = asset['parent_file_name']
            parent_fnames = [parent_fnames] if not ub.iterable(parent_fnames) else parent_fnames
            assert isinstance(parent_fnames, list)
            for fname in parent_fnames:
                name = ub.Path(fname).name
                stac_id = name.rsplit('_', 1)[0]
                stac_ids.add(stac_id)

        geom = coco_img.img['auxiliary'][0]['geos_corners']
        import kwimage
        geom = kwimage.Polygon.coerce(geom)
        gid = coco_img.img['id']

        if gid not in _cache:
            item_search = catalog.search(
                collections=[collection],
                datetime=(start, end),
                intersects=geom,
                max_items=1000,
            )
            search_iter = item_search.items()
            # result0 = next(search_iter)
            results = list(search_iter)
            result_ids = {r.id for r in results}
            _cache[gid] = results

        results = _cache[gid]
        result_ids = {r.id for r in results}
        assert result_ids.issuperset(stac_ids)
        id_to_result = {r.id: r for r in results}

        # Add multiple properties because an image may be derived from multiple
        # sources.
        stac_metadata = []
        for id in stac_ids:
            result = id_to_result[id]
            stac_obj = result.to_dict()
            stac_obj = ub.udict(stac_obj)
            small_obj = stac_obj - {'stac_extensions', 'links', 'assets', 'geometry', 'bbox', 'type', 'stac_version', 'description'}
            stac_metadata.append(small_obj)

        coco_img.img['stac_metadata'] = stac_metadata
        # coco_img.img.pop('stac_properties', None)

    # Update the CocoDataset
    from watch.utils import kwcoco_extensions
    kwcoco_extensions.reorder_video_frames(dset)
    dset.dump()
