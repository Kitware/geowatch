import ubelt as ub


class HistAccum:
    def __init__(self):
        self.accum = {}
        self.n = 0

    def update(self, data, sensor, channel):
        self.n += 1
        if sensor not in self.accum:
            self.accum[sensor] = {}
        sensor_accum = self.accum[sensor]
        if channel not in sensor_accum:
            sensor_accum[channel] = ub.ddict(lambda: 0)
        final_accum = sensor_accum[channel]
        for k, v in data.items():
            final_accum[k] += v

def checker():
    import pathlib
    a4d61fc6069144cabf8b7197b1a3472d
    dpath = pathlib.Path('/home/joncrall/data/dvc-repos/smart_watch_dvc/TA1-Uncropped-MTRA/ingress/99cdee5aec0a47cabaa434dfb8a55bdf')
    dpath = pathlib.Path('/home/joncrall/data/dvc-repos/smart_watch_dvc/TA1-Uncropped-MTRA/ingress/a4d61fc6069144cabf8b7197b1a3472d')
    for fpath in sorted(list(dpath.glob('*.tif'))):
        print('fpath.name = {!r}'.format(fpath.name))
        print(ub.cmd('gdalinfo ' + str(fpath) + ' | grep Type', shell=True)['out'].strip())


def main():
    import kwcoco
    import pickle
    import pathlib
    from watch.utils.util_data import find_smart_dvc_dpath
    from watch.utils import kwcoco_extensions

    dvc_dpath = find_smart_dvc_dpath()
    # coco_fpath = dvc_dpath / 'Drop1-Aligned-L1/data.kwcoco.json'
    coco_fpath = dvc_dpath / 'Drop1-Aligned-TA1-MTRA/data.kwcoco.json'
    coco_dset = kwcoco.CocoDataset(coco_fpath)
    bundle_dpath = pathlib.Path(coco_img.bundle_dpath)

    accum = HistAccum()
    images = coco_dset.images()
    images = images.compress([s in {'S2', 'L8'} for s in images.lookup('sensor_coarse')])

    if 0:
        for coco_img in ub.ProgIter(images.coco_images, desc='read stats'):
            sensor_coarse = coco_img.img.get('sensor_coarse', None)
            print('sensor_coarse = {!r}'.format(sensor_coarse))
            for obj in coco_img.iter_asset_objs():
                img_fpath = bundle_dpath / (obj['file_name'])
                print('img_fpath = {!r}'.format(img_fpath.name))
                print(ub.cmd('gdalinfo ' + str(img_fpath) + ' | grep Type', shell=True)['out'])
                print('img_fpath = {!r}'.format(img_fpath))

    kwcoco_extensions.coco_populate_geo_heuristics(
        coco_dset,
        gids=images, enable_intensity_stats=True, workers=12, mode='process')

    for coco_img in ub.ProgIter(images.coco_images, desc='read stats'):
        # [0:10]:
        # kwcoco_extensions.coco_populate_geo_img_heuristics2(coco_img, enable_intensity_stats=True)
        for obj in coco_img.iter_asset_objs():
            sensor = obj.get('sensor_coarse', 'unknown')
            channels = kwcoco.FusedChannelSpec.coerce(obj['channels']).as_list()
            if channels == ['cloudmask']:
                continue
            fpath = bundle_dpath / (obj['file_name'] + '.stats.pkl')
            if fpath.exists():
                with open(fpath, 'rb') as file:
                    stat_info = pickle.load(file)
                for band_idx, band_stat in enumerate(stat_info['bands']):
                    hist = band_stat['intensity_hist']
                    channel = channels[band_idx]
                    accum.update(hist, sensor, channel)

    import pandas as pd
    import numpy as np

    common = set.intersection(*[set(z) for z in accum.accum.values()]) - {'SEA4', 'SOZ4', 'SOA4', 'SEZ4'}

    common = common - {'swir16'}

    to_stack = {}
    for sensor, sub in accum.accum.items():
        for channel, hist in sub.items():
            if channel in common:
                hist = ub.sorted_keys(hist)
                hist.pop(0)
                df = pd.DataFrame({
                    'bin': np.array(list(hist.keys())),
                    'value': np.array(list(hist.values())),
                    'channel': [channel] * len(hist),
                    'sensor': [sensor] * len(hist),
                })
                to_stack[(channel, sensor)] = df

    full_df = pd.concat(list(to_stack.values()))
    full_df[~np.isfinite(full_df.bin)]
    full_df = full_df[np.isfinite(full_df.bin)]

    import kwplot
    sns = kwplot.autosns()
    # sns.lineplot(full_df=df, x='bin', y='value')
    max_val = np.iinfo(np.uint16).max
    # import kwimage
    palette = {
        # 'red': kwimage.Color('red').as01(),
        # 'blue': kwimage.Color('blue').as01(),
        # 'green': kwimage.Color('green').as01(),
        # 'cirrus': kwimage.Color('skyblue').as01(),
        # 'coastal': kwimage.Color('purple').as01(),
        # 'nir': kwimage.Color('orange').as01(),
        # 'swir16': kwimage.Color('pink').as01(),
        # 'swir22': kwimage.Color('hotpink').as01(),
        'red': 'red',
        'blue': 'blue',
        'green': 'green',
        'cirrus': 'skyblue',
        'coastal': 'purple',
        'nir': 'orange',
        'swir16': 'pink',
        'swir22': 'hotpink',
    }

    hist_data_kw = dict(
        x='bin', weights='value', bins=256, stat='count', hue='channel',
    )
    hist_style_kw = dict(
        fill=True,
        element='step',
        palette=palette,
        # multiple='stack',
        kde=True,
    )

    #  For S2 that is supposed to be divide by 10000.  For L8 it is multiply by 2.75e-5 and subtract 0.2.
    # 1 / 2.75e-5

    sensor_maxes = {
        'S2' : 6e4,
        'L8' : int(2 ** 16),
    }

    fig = kwplot.figure(fnum=1, doclf=True)
    pnum_ = kwplot.PlotNums(nSubplots=2, nRows=1)
    for sensor_name, sensor_df in full_df.groupby('sensor'):

        info_rows = []
        for channel, chan_df in sensor_df.groupby('channel'):
            info = {
                'min': chan_df.bin.min(),
                'max': chan_df.bin.max(),
                'channel': channel,
                'sensor': sensor_name,
            }
            info_rows.append(info)
        print(pd.DataFrame(info_rows))
        print('info = {!r}'.format(info))

        ax = kwplot.figure(fnum=1, pnum=pnum_()).gca()
        sns.histplot(ax=ax, data=sensor_df, **hist_data_kw, **hist_style_kw)
        ax.set_title(sensor_name)
        # maxx = sensor_df.bin.max()
        maxx = sensor_maxes[sensor_name]
        ax.set_xlim(0, maxx)

        # ax.set_xscale('symlog', base=2)
        # int(2 ** 16 - 1))

        # ax = kwplot.figure(fnum=1, pnum=(1, 2, 2)).gca()
        # sns.histplot(
        #     data=df_s2, x='bin', weights='value', bins=64, stat='count',
        #     hue='channel',
        #     # col='sensor',
        #     multiple='stack',
        #     fill=True,
        #     element='step',
        #     palette=palette,
        #     # kde=True,
        #     ax=ax,
        # )
        # ax.set_title('S2')
        # ax.set_xlim(0, int(2 ** 16 - 1))
