

def test_ignore_buffer_region():

    # Basic Data Sampling
    from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset

    import ndsampler
    import kwcoco
    import geowatch
    import ubelt as ub
    from geowatch.cli import coco_add_ignore_buffer
    import geowatch
    import ubelt as ub
    import kwcoco
    dpath = ub.Path.appdir('geowatch/tests/ignore_buffer_2')
    dpath.ensuredir()
    ignore_buffer_size = '10@10GSD'
    dst = dpath / 'out.kwcoco.zip'
    src = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True)
    kwargs = dict(src=src.data_fpath, dst=dst, ignore_buffer_size=ignore_buffer_size)
    coco_add_ignore_buffer.main(cmdline=0, **kwargs)
    coco_dset = kwcoco.CocoDataset.coerce(dst)
    sampler = ndsampler.CocoSampler(coco_dset)
    self = KWCocoVideoDataset(sampler, time_dims=4, window_dims=(300, 300),
                            channels='auto')
    self.disable_augmenter = True
    index = self.sample_grid['targets'][self.sample_grid['positives_indexes'][0]]
    item = self[index]
    # Summarize batch item in text
    summary = self.summarize_item(item)
    print('item summary: ' + ub.urepr(summary, nl=2))
    # Draw batch item
    canvas = self.draw_item(item)
    # xdoctest: +REQUIRES(--show)
    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas)
    kwplot.show_if_requested()

if __name__ == "__main__":
    test_ignore_buffer_region()

    
    