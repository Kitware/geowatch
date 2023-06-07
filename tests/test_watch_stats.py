def test_empty_stats():
    import kwcoco
    dset = kwcoco.CocoDataset()
    from watch.cli import watch_coco_stats
    watch_coco_stats.__config__.main(cmdline=0, src=[dset])
