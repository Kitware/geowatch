

def main():
    """
    Get filesize stats for a bundle, without needing the kwcoco file itself.
    """
    import pandas as pd
    import watch
    import ubelt as ub
    import xdev
    dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')

    bundle_dpath = dvc_dpath / 'Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2'
    # bundle_dpath = dvc_dpath / 'Drop4-BAS'

    subfolders = list(bundle_dpath.glob('*/*'))
    subfolders = [p for p in subfolders if 'json' not in p.name and not p.name.startswith('_')]

    if 0:
        jobs = ub.JobPool(mode='thread', max_workers=8)
        for dpath in subfolders:
            command = f'du -sL "{dpath}"'
            job = jobs.submit(ub.cmd, command)
            job.dpath = dpath
        rows = []
        for job in jobs.as_completed(desc='collect size jobs'):
            out = job.result()['out']
            size_str = out.split('\t')[0]
            job.size_str = size_str
            num_bytes = int(size_str)
            rows.append({
                'dpath': job.dpath,
                'num_bytes': num_bytes * 1024,
            })
        for row in rows:
            dpath = row['dpath']
            suffix = dpath.relative_to(bundle_dpath)
            row['region_id'] = suffix.parent.name
            row['sensor'] = suffix.name
            row['size'] = xdev.byte_str(row['num_bytes'])

        df = pd.DataFrame(rows)
        piv = df.pivot(['region_id'], ['sensor'], ['num_bytes'])
        import rich
        rich.print(piv.applymap(lambda x: x if pd.isnull(x) else xdev.byte_str(x)).to_string())
        piv.sum(axis=0).apply(xdev.byte_str)

    per_band_rows = []
    for dpath in ub.ProgIter(subfolders, desc='per-image-stats'):
        frame_dpaths = list(dpath.glob('*/*'))
        for dpath in frame_dpaths:
            suffix = dpath.relative_to(bundle_dpath)
            for fpath in dpath.ls():
                fpath.name
                stat = fpath.stat()
                per_band_rows.append({
                    'fpath': fpath,
                    'region_id': suffix.parts[0],
                    'sensor': suffix.parts[1],
                    'image_name': dpath.name,
                    'band': fpath.name.split('_')[-1].split('.')[0],
                    'is_dangling': fpath.name.startswith('.'),
                    'st_size': stat.st_size,
                    'st_mtime': stat.st_mtime,
                })
        row['num_frames'] = len(frame_dpaths)
    band_df = pd.DataFrame(per_band_rows)

    dangling_temp_files = band_df[band_df['is_dangling']]

    if 0:
        for fpath in ub.ProgIter(dangling_temp_files['fpath']):
            assert '.tmp' in fpath.name
            fpath.delete()
    regular_files = band_df[~band_df['is_dangling']]
    print(xdev.byte_str(dangling_temp_files['st_size'].sum()))
    print(xdev.byte_str(regular_files['st_size'].sum()))

    per_image_rows = []
    for _, group in band_df.groupby('image_name'):
        sensor_spec = ','.join(group['sensor'].unique())
        region_id = ','.join(group['region_id'].unique())
        chan_spec = ','.join(sorted(group['band'].to_list()))
        per_image_rows.append({
            'sensor': sensor_spec,
            'chan_spec': chan_spec,
            'region_id': region_id,
            'num_bytes': group['st_size'].sum(),
        })
    per_image_df = pd.DataFrame(per_image_rows)

    print(ub.repr2(ub.dict_hist(per_image_df['chan_spec'])))

    band_df = band_df[~band_df['is_dangling']]
    # # band_df = band_df[band_df['sensor'] == 'WV']
    # grouped_sizestr = band_df[
    #     (band_df['region_id'] == 'BR_R005') |
    #     (band_df['region_id'] == 'BR_R004')
    # ]

    groups = band_df.groupby(['region_id', 'sensor', 'band'])

    grouped_size_sum = groups['st_size'].sum()
    grouped_size_num = groups['st_size'].count()
    grouped_size_std = groups['st_size'].std()
    grouped_size_mean = groups['st_size'].mean()

    grouped_sizestr = grouped_size_sum.apply(xdev.byte_str).to_frame().reset_index()
    grouped_sizestr = grouped_sizestr[grouped_sizestr['sensor'] == 'WV']
    grouped_sizestr = grouped_sizestr[
        (grouped_sizestr['region_id'] == 'BR_R005') |
        (grouped_sizestr['region_id'] == 'BR_R004')
    ]
    piv = grouped_sizestr.pivot(['region_id', 'sensor'], ['band'], ['st_size']).T
    print(piv.to_string())

    groups = band_df.groupby(['region_id', 'sensor'])
    region_sensor_df = groups['st_size'].sum().apply(xdev.byte_str).to_frame().reset_index()
    piv = region_sensor_df.pivot(['region_id'], ['sensor'], ['st_size'])
    print(piv.to_string())

    sensor_sizes = groups['st_size'].sum().to_frame().reset_index().pivot(['region_id'], ['sensor'], ['st_size']).sum()
    sensor_sums = sensor_sizes.apply(xdev.byte_str)
    print(sensor_sums.to_string())


    df = pd.DataFrame(rows)
    df['num_frames'] = df['num_frames'].astype(pd.Int64Dtype())
    piv = df.pivot(['region'], ['sensor'], ['num_frames'])
    # piv = df.pivot(['region'], ['sensor'], ['num_bytes'])
    import rich
    rich.print(piv.to_string())

    piv.sum(axis=0).apply(xdev.byte_str)

    # Check annotation data
    region_rows = []
    from watch.utils import util_gis
    import numpy as np
    infos = list(util_gis.coerce_geojson_datas((dvc_dpath / 'annotations' / 'region_models')))
    for info in infos:
        region_data = info['data']
        region_row = region_data[region_data['type'] == 'region']
        region_row_utf = util_gis.project_gdf_to_local_utm(region_row)
        region_id = region_row['region_id'].iloc[0]
        rt_area = np.sqrt(region_row_utf['geometry'].iloc[0].area)
        region_rows.append({
            'region_id': region_id,
            'root_area_meters': rt_area,
        })

    region_df = pd.DataFrame(region_rows)
    region_df = region_df.sort_values('region_id')
    print(region_df.to_string())
