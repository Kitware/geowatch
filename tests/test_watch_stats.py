def test_empty_stats():
    import kwcoco
    dset = kwcoco.CocoDataset()
    from geowatch.cli import watch_coco_stats
    watch_coco_stats.__cli__.main(cmdline=0, src=[dset])
