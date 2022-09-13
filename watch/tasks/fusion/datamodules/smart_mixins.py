import pandas as pd
import ubelt as ub
from watch import heuristics


class SMARTDataMixin:

    def check_nested_pool(self, num=4096):
        """
        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import watch
            >>> import ndsampler
            >>> import kwcoco
            >>> dvc_dpath = watch.find_dvc_dpath()
            >>> coco_fpath = dvc_dpath / 'Cropped-Drop3-TA1-2022-03-10/data_nowv_train.kwcoco.json'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = KWCocoVideoDataset(
            >>>     sampler,
            >>>     sample_shape=(11, 256, 256),
            >>>     window_overlap=0,
            >>>     #channels="ASI|MF_Norm|AF|EVI|red|green|blue|swir16|swir22|nir",
            >>>     channels="blue|green|red|nir|swir16|swir22",
            >>>     neg_to_pos_ratio=0, time_sampling='soft2', diff_inputs=0, temporal_dropout=0.5,
            >>> )
            >>> #self.requested_tasks['change'] = False

            if 0:
                infos = []
                for num in [500, 1000, 2500, 5000, 7500, 10000, 20000]:
                    row = self.check_nested_pool(num=num)
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
        idx_hist = ub.dict_hist(self.nested_pool.sample() for _ in range(num))
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
        # NOTES ON QUALITY / CLOUDMASK
        # https://github.com/GERSL/Fmask#46-version
        # The cloudmask band is a class-idx based raster with labels
        # 0 => clear land pixel
        # 1 => clear water pixel
        # 2 => cloud shadow
        # 3 => snow
        # 4 => cloud
        # 255 => no observation

        # However, in my data I seem to see:
        # Unique values   8,  16,  65, 128

        # These are specs
        # https://smartgitlab.com/TE/standards/-/wikis/Data-Output-Specifications#quality-band
        # TODO: this could be a specially handled frame like ASI.
        quality_aliases = ['quality', 'cloudmask']
        for quality_chan_name in quality_aliases:
            if quality_chan_name in coco_img.channels:
                break
        if quality_chan_name in coco_img.channels:
            import operator as op
            import functools
            tr_cloud = tr_frame.copy()
            tr_cloud['channels'] = quality_chan_name
            # tr_cloud['channels'] = 'red|green|blue'
            tr_cloud['antialias'] = False
            tr_cloud['interpolation'] = 'nearest'
            tr_cloud['nodata'] = None
            cloud_sample = sampler.load_sample(
                tr_cloud, with_annots=None,
                padkw={'constant_values': 255},
                # dtype=np.float32
            )
            cloud_im = cloud_sample['im']
            # if tr_cloud.get('use_native_scale', None):
            # cloud_im = cloud_im[0][0]

            iffy_bits = functools.reduce(
                op.or_, ub.take(heuristics.QUALITY_BITS,
                                ['dilated_cloud', 'cirrus', 'cloud']))
            is_cloud_iffy = (cloud_im & iffy_bits) > 0
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

        print(ub.repr2(ub.dict_diff(freqs, {'gid'})))
