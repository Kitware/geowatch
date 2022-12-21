"""
Developer notebook
"""


def visualize_invariant_batch():
    import watch
    from watch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset
    dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    coco_fpath = dvc_dpath / 'Drop4-BAS/combo_vali_I2.kwcoco.json'
    self = KWCocoVideoDataset(coco_fpath,
                              time_steps=7,
                              chip_dims=(196, 196),
                              time_sampling='uniform',
                              input_space_scale='native',
                              window_space_scale='10GSD',
                              output_space_scale='10GSD',
                              channels='red|green|blue,invariants.0:3,invariants.16')
    self.disable_augmenter = True
    target = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][0]].copy()

    target['SAMECOLOR_QUALITY_HEURISTIC'] = 'region'
    target['FORCE_LOADING_BAD_IMAGES'] = 0
    target['MASK_LOW_QUALITY_PIXELS'] = 1
    target['quality_threshold'] = 0.5
    target['observable_threshold'] = 0.5
    target['resample_invalid_frames'] = 3
    item = self[target]

    #print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))
    canvas = self.draw_item(item, overlay_on_image=0, rescale=1, max_channels=5, combinable_extra='invariants.0:3')

    # xdoctest: +REQUIRES(--show)
    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas)
    kwplot.show_if_requested()


#     if 0:
#         data = item['frames'][0]['modes']['invariants.0|invariants.1|invariants.2'][..., 0].numpy()


# import timerit
# ti = timerit.Timerit(100, bestof=10, verbose=2)
# for timer in ti.reset('time'):
#     with timer:
#         # Speed up the compuation by doing this at a coarser scale
#         is_samecolor = util_kwimage.find_samecolor_regions(bands, scale=0.4, min_region_size=49)

# import timerit
# ti = timerit.Timerit(100, bestof=10, verbose=2)
# for timer in ti.reset('time'):
#     with timer:
#         util_kwimage.find_high_frequency_values(bands)
