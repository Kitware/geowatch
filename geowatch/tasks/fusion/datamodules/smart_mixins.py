import pandas as pd
import ubelt as ub


class SMARTDataMixin:

    def check_balanced_sample_tree(self, num=4096):
        """
        Developer function to check statistics about how the nested pool is
        sampling regions.

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> from geowatch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import geowatch
            >>> import ndsampler
            >>> import kwcoco
            >>> dvc_dpath = geowatch.find_dvc_dpath()
            >>> coco_fpath = dvc_dpath / 'Cropped-Drop3-TA1-2022-03-10/data_nowv_train.kwcoco.json'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = KWCocoVideoDataset(
            >>>     sampler,
            >>>     time_dims=5, window_dims=(256, 256),
            >>>     window_overlap=0,
            >>>     #channels="ASI|MF_Norm|AF|EVI|red|green|blue|swir16|swir22|nir",
            >>>     channels="blue|green|red|nir|swir16|swir22",
            >>>     neg_to_pos_ratio=0, time_sampling='soft2', diff_inputs=0, temporal_dropout=0.5,
            >>> )
            >>> #self.requested_tasks['change'] = False

            if 0:
                infos = []
                for num in [500, 1000, 2500, 5000, 7500, 10000, 20000]:
                    row = self.check_balanced_sample_tree(num=num)
                    infos.append(row)
                df = pd.DataFrame(infos)
                import kwplot
                sns = kwplot.autosns()

                data = df.melt(id_vars=['num'])
                data['style'] = 'raw'
                data.loc[data.variable.apply(lambda x: 'gids' in x), 'style'] = 'gids'
                data.loc[data.variable.apply(lambda x: 'region' in x), 'style'] = 'region'
                data['region'] = data.variable.apply(lambda x: x.split('_', 2)[-1].replace('seen', '') if 'R' in x else x)
                sns.lineplot(data=data, x='num', y='value', style='style', hue='region')

                    frac_seen = info['frac_gids_seen']
                    frac_seen['num'] = num
                    frac_seen['ideal_seen'] = ideal_seen
                    frac_seen['ideal_frac'] = ideal_frac
        """
        # Check the nested pool
        dset = self.sampler.dset
        vidid_to_name = dset.videos().lookup('name', keepid=True)
        idx_hist = ub.dict_hist(self.balanced_sample_tree.sample() for _ in range(num))
        targets = self.new_sample_grid['targets']

        gid_freq = ub.ddict(lambda: 0)
        vidid_freq = ub.ddict(lambda: 0)
        region_seen_gids = ub.ddict(set)
        for idx, freq in ub.ProgIter(list(idx_hist.items())):
            target = targets[idx]
            gids = target['gids']
            for gid in gids:
                # frame_index = dset.index.imgs[gid]['frame_index']
                gid_freq[gid] += freq
            vidid = target['video_id']
            vidname = vidid_to_name[vidid]
            region = self.vidname_to_region_name[vidname]
            vidid_freq[vidid] += freq
            region_seen_gids[region].update(gids)

        vidname_to_freq = ub.map_keys(vidid_to_name, vidid_freq)

        # TODO: these should be some concept of video groups
        region_freq = ub.ddict(lambda: 0)
        for vidname, freq in vidname_to_freq.items():
            region_name = self.vidname_to_region_name[vidname]
            region_freq[region_name] += freq

        _region_total_gids = ub.ddict(lambda: 0)
        for vidid, gids in dset.index.vidid_to_gids.items():
            vidname = vidid_to_name[vidid]
            region_name = self.vidname_to_region_name[vidname]
            _region_total_gids[region_name] += len(gids)
        region_total_num_gids = pd.Series(_region_total_gids)
        region_seen_num_gids = pd.Series(ub.map_vals(len, region_seen_gids))

        frac_gids_seen = region_seen_num_gids / region_total_num_gids

        _count = pd.Series(region_freq)
        _prob = _count / _count.sum()
        seen_gids = set(gid_freq.keys())
        total_gids = set(dset.images())
        num_seen = len(seen_gids)
        num_total = len(total_gids)
        ideal_seen = (num * len(target['gids']))
        seen_frac = num_seen / num_total
        ideal_frac = min(ideal_seen / num_total, 1.0)

        row = frac_gids_seen.add_prefix('frac_gids_seen')
        row = pd.concat([row, _prob.add_prefix('region_freq_')])

        row['seen_frac'] = seen_frac
        row['ideal_frac'] = ideal_frac
        row['num'] = num
        return row

        if 0:
            rows = []
            for idx, freq in ub.ProgIter(list(idx_hist.items())):
                target = targets[idx]
                for gid in target['gids']:
                    vidid = target['video_id']
                    vidname = vidid_to_name[vidid]
                    region = self.vidname_to_region_name[vidname]
                    frame_index = dset.index.imgs[gid]['frame_index']
                    rows.append({
                        'idx': idx,
                        'gid': gid,
                        'vidid': vidid,
                        'frame_index': frame_index,
                        'vidname': vidname,
                        'region': region,
                        'freq': freq,
                    })
            df = pd.DataFrame(rows)
            region_freq = df.groupby('region')['freq'].sum()
            region_freq = region_freq / region_freq.sum()
            _freq = df.groupby('video_id')['freq'].sum()
            _freq = _freq / _freq.sum()
            _freq = df.groupby(['video_id', 'frame_index'])['freq'].sum()
            _freq = _freq / _freq.sum()

    def _interpret_quality_mask(self, sampler, coco_img, tr_frame):
        """
        Construct a binary good/bad mask from the quality band in a coco image.

        Ignore:
            >>> from geowatch.mlops.smart_pipeline import *  # NOQA
            >>> from geowatch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import kwcoco
            >>> import ndsampler
            >>> import geowatch
            >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
            >>> dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data')
            >>> coco_fpath = dvc_dpath / 'Drop6/data_vali_split1.kwcoco.zip'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = KWCocoVideoDataset(
            >>>     sampler,
            >>>     time_dims=5, window_dims=(128, 128),
            >>>     window_overlap=0,
            >>>     channels="(S2,L8):blue|green|red|nir|cloudmask",
            >>>     input_space_scale='30GSD',
            >>>     window_space_scale='30GSD',
            >>>     output_space_scale='30GSD',
            >>>     dist_weights=1,
            >>>     use_cloudmask=1,
            >>>     resample_invalid_frames=0,
            >>>     neg_to_pos_ratio=0, time_sampling='soft2+distribute',
            >>> )
            >>> self.requested_tasks['change'] = False
            >>> target = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][0]]
            >>> gid = target['gids'][2]
            >>> tr_frame = target.copy()
            >>> tr_frame['gids'] = [gid]
            >>> coco_img = self.sampler.dset.coco_image(gid)

            import kwplot
            kwplot.autoplt()
            if 1:
                dset = self.sampler.dset
                target['resample_invalid_frames'] = 1
                target['FORCE_LOADING_BAD_IMAGES'] = 1
                item = self.getitem(target)
                canvas = self.draw_item(item)
                kwplot.imshow(canvas)

            # >>> print('item summary: ' + ub.urepr(self.summarize_item(item), nl=3))
            # >>> # xdoctest: +REQUIRES(--show)
            # >>> canvas = self.draw_item(item, max_channels=10, overlay_on_image=0, rescale=0)
            # >>> import kwplot
            # >>> kwplot.autompl()
            # >>> kwplot.imshow(canvas)
            # >>> kwplot.show_if_requested()

            tr_frame = target.copy()
            tr_frame['gids'] = [gid]

            tr_cloud = tr_frame.copy()
            quality_chan_name = 'cloudmask'
            tr_cloud['channels'] = quality_chan_name
            tr_cloud['antialias'] = False
            tr_cloud['interpolation'] = 'nearest'
            tr_cloud['nodata'] = None
            qa_sample = sampler.load_sample(
                tr_cloud, with_annots=None,

                # padkw={'constant_values': 255},
                # dtype=np.float32
            )
            quality_im = qa_data = qa_sample['im'][0]

            tr_rgb = tr_frame.copy()
            tr_rgb['channels'] = 'red|green|blue'
            rgb_sample = sampler.load_sample(
                tr_rgb, with_annots=None,
                padkw={'constant_values': 255},
                # dtype=np.float32
            )
            rgb_data = rgb_sample['im'][0]
            rgb_im = kwimage.normalize_intensity(rgb_data)
            kwplot.imshow(rgb_im)

            sensor = coco_img.img.get('sensor_coarse')
            spec_name = 'ACC-1'
            from geowatch.tasks.fusion.datamodules.qa_bands import QA_SPECS
            table = qa_table = QA_SPECS.find_table(spec_name, sensor)

            draw_cloudmask_viz(qa_data, rgb_data)

        """
        from geowatch.tasks.fusion.datamodules.qa_bands import QA_SPECS

        registered_channels = coco_img.channels

        if registered_channels is None:
            return None

        quality_aliases = ['quality', 'cloudmask']
        for quality_chan_name in quality_aliases:
            if quality_chan_name in registered_channels:
                break

        if quality_chan_name in registered_channels:
            tr_cloud = tr_frame.copy()
            tr_cloud['channels'] = quality_chan_name
            tr_cloud['antialias'] = False
            tr_cloud['interpolation'] = 'nearest'
            tr_cloud['nodata'] = None
            qa_sample = sampler.load_sample(
                tr_cloud, with_annots=None,
                # TODO: use a better constant value
                padkw={'constant_values': 0},
                # dtype=np.float32
            )
            qa_data = qa_sample['im']

            # TODO: we need a better way to map from a quality band to the
            # quality spec that it should be using.
            spec_name = 'ACC-1'
            sensor = coco_img.img.get('sensor_coarse', '*')
            try:
                table = QA_SPECS.find_table(spec_name, sensor)
            except AssertionError as ex:
                print(f'warning ex={ex}')
                is_cloud_iffy = None
            else:
                iffy_qa_names = ['cloud']
                is_cloud_iffy = table.mask_any(qa_data, iffy_qa_names)

        else:
            is_cloud_iffy = None
        return is_cloud_iffy

    def _input_grid_stats(self):
        targets = self.new_sample_grid['targets']

        freqs = ub.ddict(lambda: ub.ddict(lambda: 0))

        for target in ub.ProgIter(targets, desc='loop over targets'):
            vidid = target['video_id']
            freqs['vidid'][vidid] += 1
            gids = target['gids']
            for gid in gids:
                freqs['gid'][gid] += 1
            freqs['label'][target['label']] += 1

        dset = self.sampler.dset
        for gid, freq in freqs['gid'].items():
            sensor_coarse = dset.coco_image(gid).img.get('sensor_coarse', '*')
            sensor_coarse = dset.coco_image(gid).img.get('sensor_coarse', '*')
            freqs['sensor'][sensor_coarse] += 1

        print(ub.urepr(ub.dict_diff(freqs, {'gid'})))


def draw_cloudmask_viz(qa_data, rgb_data):
    """
    Helper visualization
    """

    import kwimage
    import numpy as np

    qabits_to_count = ub.dict_hist(qa_data.ravel())

    # For the QA band lets assign a color to each category
    colors = kwimage.Color.distinct(len(qabits_to_count))
    qabits_to_color = dict(zip(qabits_to_count, colors))

    # Colorize the QA bands
    colorized = np.empty(qa_data.shape[0:2] + (3,), dtype=np.float32)
    for qabit, color in qabits_to_color.items():
        mask = qa_data[:, :, 0] == qabit
        colorized[mask] = color

    rgb_canvas = kwimage.normalize_intensity(rgb_data)

    # Because the QA band is categorical, we should be able to make a short

    qa_canvas = colorized
    import kwplot
    legend = kwplot.make_legend_img(qabits_to_color)  # Make a legend

    # Stack things together into a nice single picture
    qa_canvas = kwimage.stack_images([qa_canvas, legend], axis=1)
    canvas = kwimage.stack_images([rgb_canvas, qa_canvas], axis=1)
    import kwplot
    kwplot.imshow(canvas)
