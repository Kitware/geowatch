import pathlib
import ubelt as ub
import numpy as np
import kwimage
from watch.tasks.fusion.datamodules.kwcoco_video_data import KWCocoVideoDataset, lookup_track_info  # NOQA


def _draw_tracks():
    import ndsampler
    import kwcoco
    from watch.utils.util_data import find_smart_dvc_dpath
    dvc_dpath = find_smart_dvc_dpath()
    coco_fpath = dvc_dpath / 'drop1-S2-L8-aligned/combo_data.kwcoco.json'
    coco_dset = kwcoco.CocoDataset(coco_fpath)
    sampler = ndsampler.CocoSampler(coco_dset)

    channel_groups = [
        'blue|green|red',
        'nir|swir16|swir22',
        'inv_shared1|inv_shared2|inv_shared3',
        'inv_shared4|inv_shared5|inv_shared6',
        'matseg_0|matseg_1|matseg_2',
        'matseg_3|matseg_4|matseg_5',
        'ASI|BSI|MBI',

        'bare_ground|built_up|cropland',
        'inland_water|snow_or_ice_field|sebkha',
        # 'forest_deciduous|forest_evergreen'
        # |brush|grassland|bare_ground|built_up|cropland|rice_field|marsh|swamp|inland_water|snow_or_ice_field|reef|sand_dune|sebkha|ocean<10m|ocean>10m|lake|river|beach|alluvial_deposits|med_low_density_built_up
    ]
    channels = '|'.join(channel_groups)

    combinable_extra = []
    for group in channel_groups[1:]:
        combinable_extra.append(list(group.split('|')))

    self = KWCocoVideoDataset(sampler, sample_shape=None, channels=channels, mode='custom', diff_inputs=True)
    self.disable_augmenter = True
    self.with_change = False

    # vidids = list(coco_dset.index.videos.keys())
    tids = list(coco_dset.index.trackid_to_aids.keys())

    dump_dpath = pathlib.Path(ub.ensuredir('./trackviz-2021-10-20'))
    tids = [35]
    for tid in tids:
        track_info = lookup_track_info(coco_dset, tid)

        member_aid = ub.peek(coco_dset.index.trackid_to_aids[tid])
        member_ann = coco_dset.index.anns[member_aid]
        member_img = coco_dset.index.imgs[member_ann['image_id']]
        vidid = member_img['video_id']
        vidname = coco_dset.index.videos[vidid]['name']

        print('tid = {!r}'.format(tid))
        dump_fpath = dump_dpath / f'video{vidid:04d}_{vidname}_track{tid:04d}.jpg'
        gids = track_info['track_gids']

        vidspace_box = track_info['full_vid_box'].scale(1.9, about='center')

        idxs = np.unique(np.linspace(0, len(gids) - 1, 17).round().astype(int))
        chosen_gids = np.array(gids)[idxs]

        index = {
            'space_slice': vidspace_box.quantize().to_slices()[0],
            'gids': chosen_gids,
            'video_id': vidid,
        }
        # img = coco_dset.imgs[gids[0]]

        item = self.__getitem__(index)

        # if 0:
        #     from skimage import exposure
        #     from skimage.exposure import match_histograms
        #     references = {}
        #     for frame in item['frames']:
        #         for mode_key, mode_val in frame['modes'].items():
        #             reference = references.get(mode_key)
        #             if reference is None:
        #                 references[mode_key] = mode_val
        #             else:
        #                 stack = []
        #                 for ref_chan, other_chan in zip(reference, mode_val):
        #                     ref_np = ref_chan.numpy()
        #                     other_np = other_chan.numpy()

        #                     # min_ = min(other_chan.max(), ref_chan.max())
        #                     # max_ = min(other_chan.min(), ref_chan.min())
        #                     # extent = max(max_ - min_, 1e-8)
        #                     # sf = (2 ** 32) / extent
        #                     # other_quant = (other_chan.numpy() * sf).astype(np.int32)
        #                     # ref_quant = (ref_chan.numpy() * sf).astype(np.int32)
        #                     new_other = match_histograms(other_np, ref_np)
        #                     stack.append(new_other)
        #                 new_mode = np.stack(stack, axis=0)
        #                 frame['modes'][mode_key] = torch.Tensor(new_mode)

        canvas = self.draw_item(item, combinable_extra=combinable_extra,
                                max_dim=384, overlay_on_image=False,
                                norm_over_time=1, max_channels=7)

        if 1:
            import kwplot
            kwplot.autompl()
            kwplot.imshow(canvas)
            kwplot.show_if_requested()
            break
        else:
            print('dump_fpath = {!r}'.format(dump_fpath))
            kwimage.imwrite(str(dump_fpath), canvas)
        # xdoctest: +REQUIRES(--show)
