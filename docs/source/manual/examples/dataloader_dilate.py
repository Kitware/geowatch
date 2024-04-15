

def demo_dataloader_dilate():
    import geowatch
    from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    coco_dset = geowatch.coerce_kwcoco('vidshapes1', num_frames=10)
    self = KWCocoVideoDataset(coco_dset, time_dims=5,
                              window_dims=(512, 512),
                              balance_areas=True,
                              dist_weights=True,
                              normalize_perframe=False)
    self.disable_augmenter = True

    # Choose the target region we will sample
    target = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][0]]

    target['dist_weights'] = True
    item1 = self[target]

    target['dist_weights'] = False
    item2 = self[target]

    # Visualize the item
    canvas1 = self.draw_item(item1)
    canvas2 = self.draw_item(item2)

    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas1, pnum=(1, 2, 1), fnum=1, title='dist_weights=1')
    kwplot.imshow(canvas2, pnum=(1, 2, 2), fnum=1, title='dist_weights=0')
    kwplot.show_if_requested()
