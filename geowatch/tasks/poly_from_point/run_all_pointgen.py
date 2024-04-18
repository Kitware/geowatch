def main():
    """
    Developer script to get point-based polygons generated.
    Needs cleanup.
    """
    import cmd_queue
    import ubelt as ub

    import geowatch
    data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase3_data', hardware='ssd')
    # expt_dvc_dpath = geowatch.find_dvc_dpath(tags='phase3_expt', hardware='auto')

    kwcoco_dpath = (data_dvc_dpath / 'Drop8-ARA-Median10GSD-V1')

    region_dpath = ((data_dvc_dpath / 'annotations') / 'drop8-v1/region_models')
    points_fpath = data_dvc_dpath / 'annotations/point_based_annotations.zip'

    if 1:
        # Based on T&E submodule
        points_fpath = data_dvc_dpath / 'submodules/annotations/supplemental_data/point_based_annotations.geojson'
        region_dpath = data_dvc_dpath / 'submodules/annotations/region_models/'

    assert points_fpath.exists()

    import geopandas as gpd
    # Read json text directly from the zipfile
    if points_fpath.endswith('.zip'):
        file = ub.zopen(points_fpath + "/" + "point_based_annotations.geojson", "r")
    else:
        file = open(points_fpath, 'r')
    with file:
        points_gdf_crs84 = gpd.read_file(file)

    points_gdf_crs84['region_id'] = points_gdf_crs84['site_id'].apply(lambda s: '_'.join(s.split('_')[0:2]))
    regions_with_points = points_gdf_crs84['region_id'].unique()

    common_params = dict(
        size_prior='20.06063 x 20.0141229 @ 10mGSD',
        ignore_buffer=None,
        time_prior='1year',
        filepath_to_points=points_fpath,
    )

    dest_region_dpath = ((data_dvc_dpath / 'annotations') / 'drop8-points-v1/region_models').ensuredir()
    viz_region_dpath = ((data_dvc_dpath / 'annotations') / 'drop8-points-v1/region_models_viz').ensuredir()

    polygen_template = ub.codeblock(
        r"""
        python -m geowatch.tasks.poly_from_point.predict \
            --method 'ellipse' \
            --region_id "{region_id}" \
            --filepath_to_images "{filepath_to_images}" \
            --filepath_to_points "{filepath_to_points}" \
            --filepath_to_region "{filepath_to_region}" \
            --time_prior "{time_prior}" \
            --ignore_buffer "{ignore_buffer}" \
            --size_prior "{size_prior}" \
            --filepath_output "{filepath_output}"
        """
    )

    regionviz_template = ub.codeblock(
        r"""
        python -m geowatch.tasks.poly_from_point.predict \
            --method 'ellipse' \
            --region_id "{region_id}" \
            --filepath_to_images "{filepath_to_images}" \
            --filepath_to_points "{filepath_to_points}" \
            --filepath_to_region "{filepath_to_region}" \
            --time_prior "{time_prior}" \
            --ignore_buffer "{ignore_buffer}" \
            --size_prior "{size_prior}" \
            --filepath_output "{filepath_output}"
        """
    )
    region_model_list = list(region_dpath.glob("*.geojson"))

    queue = cmd_queue.Queue.create(backend="tmux", size=4)
    # queue.add_header_command(
    #     ub.codeblock(
    #         """
    #         pyenv shell 3.10.5
    #         source $(pyenv prefix)/envs/pyenv-geowatch/bin/activate
    #         """
    #     )
    # )

    regionviz_template = ub.codeblock(
        '''
        geowatch draw_region "{region_fpath}" --fpath "{viz_fpath}"
        ''')

    status_rows = []
    for region_path in region_model_list:
        region_id = region_path.stem
        kwcoco_path = (
            kwcoco_dpath / region_id / (f"imgonly-{region_id}-rawbands.kwcoco.zip")
        )
        has_points = region_id in regions_with_points
        has_kwcoco = kwcoco_path.exists()
        filepath_output = dest_region_dpath / f'{region_id}.geojson'
        row = {
            'region_id': region_id,
            'annotation_type': region_id.split('_')[1][0],
            'has_points': has_points,
            'has_kwcoco': has_kwcoco,
        }
        if has_kwcoco and has_points:
            fmtkw = {
                "region_id": region_id,
                "filepath_to_images": None,
                "filepath_to_region": region_path,
                "filepath_output": filepath_output,
                **common_params
            }
            polygen_cmd = polygen_template.format(**fmtkw)
            polygen_job = queue.submit(
                polygen_cmd, name=f"PolyGen: {region_id} ")

            DO_DRAW = 1
            if DO_DRAW:
                viz_fpath = viz_region_dpath / f'point_viz_{region_id}.png'
                regionviz_cmd = regionviz_template.format(
                    region_fpath=filepath_output,
                    viz_fpath=viz_fpath,
                )
                queue.submit(regionviz_cmd, name=f'Viz: {region_id}',
                             depends=polygen_job)
        status_rows.append(row)

    import pandas as pd
    status_df = pd.DataFrame(status_rows)
    status_df.value_counts('has_kwcoco')
    # rows_to_process = status_df[status_df.has_kwcoco & status_df.has_points]
    # rows_to_process['annotation_type'].value_counts()
    # status_df[~status_df['has_kwcoco']]

    queue.print_graph()
    queue.print_commands()
    queue.run()


if __name__ == "__main__":
    main()
