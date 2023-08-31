import kwcoco
from kwutil import util_progress
import math
# from watch.cli.coco_spectra import _weighted_auto_bins, _fill_missing_colors
from watch.cli.coco_spectra import ensure_intensity_stats, HistAccum
import ubelt as ub


def main():
    dset = kwcoco.CocoDataset('Aligned-Drop7/KR_R001/imgonly_KR_R001_cold-L8S2-HTR.kwcoco.zip')

    cold_imgs = []
    for coco_img in dset.images().coco_images:
        if 'red_COLD_a1' in coco_img.channels:
            cold_imgs.append(coco_img)

    l8_imgs = [c for c in cold_imgs if c['sensor_coarse'] == 'L8']
    # for c in l8_imgs:
    #     delayed = c.imdelay('red_COLD_a1')
    #     data = delayed.finalize()

    # sns.histplot(ax=ax, data=sensor_df.reset_index(), **hist_data_kw_, **hist_style_kw)

    exclude_channels = None

    # green_COLD_cv,green_COLD_rmse,green_COLD_a0,green_COLD_a1,green_COLD_b1,green_COLD_c1,red_COLD_cv,red_COLD_rmse,red_COLD_a0,red_COLD_a1,red_COLD_b1,red_COLD_c1,nir_COLD_cv,nir_COLD_rmse,nir_COLD_a0,nir_COLD_a1,nir_COLD_b1,nir_COLD_c1,swir16_COLD_cv,swir16_COLD_rmse,swir16_COLD_a0,swir16_COLD_a1,swir16_COLD_b1,swir16_COLD_c1,swir22_COLD_cv,swir22_COLD_rmse,swir22_COLD_a0,swir22_COLD_a1,swir22_COLD_b1,swir22_COLD_c1

    include_channels = ['blue_COLD_cv', 'blue_COLD_rmse', 'blue_COLD_a0', 'blue_COLD_a1', 'blue_COLD_b1', 'blue_COLD_c1']

    jobs = ub.JobPool(mode='process', max_workers=16, transient=True)
    pman = util_progress.ProgressManager()
    with pman:
        for coco_img in pman.progiter(l8_imgs, desc='submit stats jobs'):
            coco_img.detach()
            job = jobs.submit(ensure_intensity_stats, coco_img, recompute=True,
                              include_channels=include_channels,
                              exclude_channels=exclude_channels)
            job.coco_img = coco_img

        valid_min = -math.inf
        valid_max = math.inf
        # valid_min = 1
        # valid_max = 2000

        date_to_accum = {}
        for job in pman.progiter(jobs.as_completed(), total=len(jobs), desc='accumulate stats'):
            intensity_stats = job.result()
            sensor = job.coco_img.get('sensor_coarse', job.coco_img.get('sensor', 'unknown_sensor'))
            date = job.coco_img.get('date_captured')
            if date not in date_to_accum:
                date_to_accum[date] = HistAccum()
            accum = date_to_accum[date]
            for band_stats in intensity_stats['bands']:
                intensity_hist = band_stats['intensity_hist']
                band_props = band_stats['properties']
                band_name = band_props['band_name']
                intensity_hist = {k: v for k, v in intensity_hist.items()
                                  if k >= valid_min and k <= valid_max}
                accum.update(intensity_hist, sensor, band_name)

    date_to_df = {}
    for date, accum in date_to_accum.items():
        full_df = accum.finalize()
        date_to_df[date] = full_df

    import kwplot
    kwplot.autosns()
    from watch.cli.coco_spectra import plot_intensity_histograms
    from watch.cli.coco_spectra import CocoSpectraConfig
    config = CocoSpectraConfig()
    # config['valid_range'] = '1:2000'

    import kwplot
    from kwutil.util_time import coerce_datetime
    kwplot.figure(fnum=1, doclf=True)
    pnum_ = kwplot.PlotNums(nSubplots=len(date_to_df))
    for date, full_df in ub.udict(date_to_df).sorted_keys().items():
        dt = coerce_datetime(date)
        sensor = full_df.iloc[0]['sensor']
        ax = kwplot.figure(fnum=1, pnum=pnum_()).gca()
        plot_intensity_histograms(full_df, config, ax=ax)
        ax.set_title(sensor + ' ' + dt.date().isoformat())
        ax.set_ylim(0, 0.01)
        ax.set_xlim(-500, 1500)
        # ax.set_yscale('symlog')
