
def run_diagnostics():
    import watch
    import ubelt as ub
    import kwcoco

    import kwplot
    kwplot.autompl(force='Agg')

    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
    bundle_dpath = dvc_dpath / 'Drop1-Aligned-TA1-2022-01'
    coco_fpath = bundle_dpath / 'data.kwcoco.json'

    dset_name = bundle_dpath.stem

    analytic_dpath = ub.Path(bundle_dpath / 'analytics').ensuredir()

    coco_dset = kwcoco.CocoDataset.coerce(coco_fpath)

    # Python variant of watch-stats
    # TODO: return nice results?
    from watch.cli import watch_coco_stats
    watch_coco_stats.WatchCocoStats.main(src=[coco_dset])

    videos = coco_dset.videos()

    channels = 'coastal|blue|green|red|nir|swir16|swir22'

    # Python variant of coco_intensity_histograms
    from watch.cli import coco_intensity_histograms
    kwargs = dict(
        src=coco_dset,
        dst=analytic_dpath / 'intensity_hist.png',
        include_channels=channels,
        title=dset_name,
        valid_range='1:5000',
        bins=512)
    coco_intensity_histograms.main(**kwargs)

    for vidid in ub.ProgIter(videos, verbose=3):
        region_name = coco_dset.videos([vidid]).lookup('name')[0]

        region_intensity_dpath = (analytic_dpath / 'regions_intensity').ensuredir()
        hist_fpath = region_intensity_dpath / f'intensity_hist_{region_name}.png'

        sub_gids = coco_dset.images(vidid=vidid)
        region_coco = coco_dset.subset(sub_gids)

        coco_intensity_histograms.main(
            src=region_coco,
            dst=hist_fpath,
            title=f'{dset_name}: {region_name}',
            include_channels=channels,
            valid_range='1:5000',
            bins=512,
        )
