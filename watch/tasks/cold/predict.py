"""
Proof of concept that runs PYCOLD on a kwcoco file

Unfinished.

TODO:
    - [x] Convert a kwcoco dataset to the temporal block format
    - [x] Run pycold on the block
    - [ ] Summarize the pycold results and reconstruct rasters corresponding to the original image
    - [ ] Write results out as a new kwcoco file suitable for fusion.

Notes:
    Relies on the dev/kwcoco_ingest of pycold: git@github.com:GERSL/pycold
"""


def main():
    """
    """
    from pycold import cold_detect
    from pycold.utils import read_blockdata
    import ubelt as ub
    import logging
    import kwcoco
    logging.basicConfig(level='INFO')

    # Ideally we can swap this out for a bigger DVC dataset.
    if 0:
        import watch
        dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        coco_fpath = dvc_dpath / 'Drop4-BAS/data_vali.kwcoco.json'
        coco_dset = kwcoco.CocoDataset.coerce(coco_fpath)
    else:
        # For now use the demo kwcoco dataset
        from pycold.imagetool.prepare_kwcoco import grab_demo_kwcoco_dataset
        coco_fpath = grab_demo_kwcoco_dataset()
        coco_dset = kwcoco.CocoDataset.coerce(coco_fpath)

    # Construct a unique hashid to make a temporary directory
    # specific to this dataset.
    dset_hashid = coco_dset._cached_hashid()[0:8]

    # Prepare the kwcoco data
    dpath = (ub.Path.appdir('pycold/kwcoco_prep') / dset_hashid).ensuredir()
    out_dir = (dpath / 'stacked').ensuredir()
    config, results = stack_kwcoco(coco_fpath, out_dir)  # NOQA

    config.update({
        # hack
        'n_cols': 2,
        'n_rows': 2,
    })

    from pycold.ob_analyst import ObjectAnalystHPC
    from datetime import datetime as datetime_cls
    start_dt = datetime_cls(2012, 1, 1, 0, 0, 0)
    starting_date = start_dt.toordinal()

    # For more details of how ObjectAnalysistHPC is used
    # see: ~/code/pycold/src/python/pycold/imagetool/tile_processing.py
    # We need to see how we can apply it here.
    result_path = (dpath / 'results').ensuredir()
    analyst = ObjectAnalystHPC(config, out_dir, result_path, starting_date)  # NOQA

    # Given the prepared data, run pycold and reconstruct
    # a feature map for each image
    video_dpaths = list(out_dir.glob('*'))
    for video_dpath in video_dpaths:

        for block_folder in video_dpath.glob('block_*'):
            # FIXME:
            # These magic numbers are determined by config used to split the data into
            # blocks. This data should be written to a manifest file that registers the
            # blocks we have written so the reading process can read the information
            # instead of assuming the developer knows that information a-priori.
            total_pixels = 6
            total_bands = 8

            img_stack, dates = read_blockdata(block_folder, total_pixels, total_bands)

            for pixel in img_stack:
                blues, greens, reds, nirs, swir1s, swir2s, thermals, qas = pixel
                try:
                    cold_result = cold_detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas)
                except Exception as ex:
                    assert 'no change records' in str(ex)
                else:
                    for i in range(len(cold_result)):
                        cold_result[i]['t_start']
                        cold_result[i]['t_end']
                        cold_result[i]['coefs']
                        cold_result[i]['t_break']
                        cold_result[i]['num_obs']
                        cold_result[i]['category']
                        cold_result[i]['change_prob']
                        cold_result[i]['rmse']
                        cold_result[i]['rmse']
                        cold_result[i]['magnitude']


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.tasks.cold.predict
    """
    main()
