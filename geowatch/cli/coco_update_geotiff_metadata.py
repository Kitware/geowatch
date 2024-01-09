import ubelt as ub
import scriptconfig as scfg
# import numpy as np
# import kwimage


class UpdateGeotiffMetadataConfig(scfg.DataConfig):
    """
    Simplified version of coco-add-watch-fields that only ensures that geotiff
    metadata is propogated to all auxiliary assets within an image.

    Modifies the underlying images on disk.  Does not modify the kwcoco file.
    """
    __default__ = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'include_sensors': scfg.Value(None, help='if specified can be comma separated valid sensors'),

        'exclude_sensors': scfg.Value(None, help='if specified can be comma separated invalid sensors'),

        'select_images': scfg.Value(
            None, type=str, help=ub.paragraph(
                '''
                A json query (via the jq spec) that specifies which images
                belong in the subset. Note, this is a passed as the body of
                the following jq query format string to filter valid ids
                '.images[] | select({select_images}) | .id'.

                Examples for this argument are as follows:
                '.id < 3' will select all image ids less than 3.
                '.file_name | test(".*png")' will select only images with
                file names that end with png.
                '.file_name | test(".*png") | not' will select only images
                with file names that do not end with png.
                '.myattr == "foo"' will select only image dictionaries
                where the value of myattr is "foo".
                '.id < 3 and (.file_name | test(".*png"))' will select only
                images with id less than 3 that are also pngs.
                .myattr | in({"val1": 1, "val4": 1}) will take images
                where myattr is either val1 or val4.

                Requries the "jq" python library is installed.
                ''')),

        'select_videos': scfg.Value(
            None, help=ub.paragraph(
                '''
                A json query (via the jq spec) that specifies which videos
                belong in the subset. Note, this is a passed as the body of
                the following jq query format string to filter valid ids
                '.videos[] | select({select_images}) | .id'.

                Examples for this argument are as follows:
                '.name | startswith("foo")' will select only videos
                where the name starts with foo.

                Only applicable for dataset that contain videos.

                Requries the "jq" python library is installed.
                ''')),

        'mode': scfg.Value('thread', help='can be thread, process, or serial'),

        'workers': scfg.Value(0, type=str, help='number of io threads'),
    }


def main(cmdline=True, **kwargs):
    """
    Ignore:
        from geowatch.cli.coco_update_geotiff_metadata import *  # NOQA
        import geowatch
        dvc_dpath = geowatch.find_dvc_dpath()
        base_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/combo_LM.kwcoco.json'
        base_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/dzyne_landcover.kwcoco.json'
        kwargs = {'src': base_fpath}
    """
    config = UpdateGeotiffMetadataConfig.cli(data=kwargs, cmdline=cmdline,
                                             strict=True)

    from geowatch.utils import kwcoco_extensions
    import kwcoco
    coco_dset = kwcoco.CocoDataset.coerce(config['src'])

    valid_gids = kwcoco_extensions.filter_image_ids(
        coco_dset,
        include_sensors=config['include_sensors'],
        exclude_sensors=config['exclude_sensors'],
        select_images=config['select_images'],
        select_videos=config['select_videos'],
    )

    from kwutil import util_parallel
    workers = util_parallel.coerce_num_workers(config['workers'])

    mode = config['mode']
    assert mode == 'thread', 'must be thread for now'
    jobs = ub.JobPool(mode=mode, max_workers=workers)

    for gid in ub.ProgIter(valid_gids, desc='submit jobs'):
        coco_img = coco_dset.coco_image(gid).detach()
        job = jobs.submit(kwcoco_extensions.transfer_geo_metadata2, coco_img, dry=0)
        job.gid = gid

    transfer_tasks = []
    failed = []
    for job in jobs.as_completed(desc='collecting geotiff transfer jobs'):
        try:
            tasks = job.result()
            transfer_tasks.append(tasks)
        except Exception as ex:
            print('Failed ex = {!r}'.format(ex))
            failed.append(job.gid)

    # all_tasks = list(ub.flatten(transfer_tasks))
    # print summary of what happpened
    tasks_per_img = list(map(len, transfer_tasks))
    import kwarray
    tasks_per_img_stats = kwarray.stats_dict(tasks_per_img, sum=True)
    print('transfers_per_img_stats = {}'.format(ub.urepr(tasks_per_img_stats, nl=1)))
