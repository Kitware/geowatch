#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class RunAllPointgenCLI(scfg.DataConfig):
    method = scfg.Value("ellipse", help="pointgen method")
    ignore_buffer = scfg.Value("20@10GSD", help="kwcoco ignore buffer size")
    size_prior = scfg.Value("20.06063 x 20.0141229 @ 10mGSD")
    time_prior = scfg.Value("1year")


def main(cmdline=1, **kwargs):
    """
    Developer script to get point-based polygons generated.
    Needs cleanup.

    Ignore:
        DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=auto)
        DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
        echo "$DVC_DATA_DPATH"
        echo "$DVC_EXPT_DPATH"
    """
    import rich
    from rich.markup import escape

    config = RunAllPointgenCLI.cli(cmdline=1, strict=True)
    rich.print("config = " + escape(ub.urepr(config, nl=1)))

    import cmd_queue
    import geowatch

    data_dvc_dpath = geowatch.find_dvc_dpath(tags="phase3_data", hardware="ssd")
    # expt_dvc_dpath = geowatch.find_dvc_dpath(tags='phase3_expt', hardware='auto')

    kwcoco_bundle_dpath = data_dvc_dpath / "Drop8-ARA-Median10GSD-V1"

    points_fpath = data_dvc_dpath / "annotations/point_based_annotations.zip"
    region_dpath = (data_dvc_dpath / "annotations") / "drop8-v1/region_models"
    empty_region_dpath = (
        data_dvc_dpath / "annotations"
    ) / "drop8-v1/empty_region_models"

    if 0:
        # Based on T&E submodule
        points_fpath = (
            data_dvc_dpath
            / "submodules/annotations/supplemental_data/point_based_annotations.geojson"
        )
        region_dpath = data_dvc_dpath / "submodules/annotations/region_models/"
        empty_region_dpath = (
            data_dvc_dpath / "submodules/annotations/empty_region_models/"
        )

    assert points_fpath.exists()

    import geopandas as gpd

    # Read json text directly from the zipfile
    if points_fpath.endswith(".zip"):
        file = ub.zopen(points_fpath + "/" + "point_based_annotations.geojson", "r")
    else:
        file = open(points_fpath, "r")
    with file:
        points_gdf_crs84 = gpd.read_file(file)

    points_gdf_crs84["region_id"] = points_gdf_crs84["site_id"].apply(
        lambda s: "_".join(s.split("_")[0:2])
    )
    regions_with_points = sorted(points_gdf_crs84["region_id"].unique())

    common_params = dict(
        size_prior=config.size_prior,
        time_prior=config.time_prior,
        filepath_to_points=points_fpath,
    )

    # TODO: do a hash of the config in this output, or just specify it in config?
    dest_annot_dpath = (
        data_dvc_dpath / "annotations"
    ) / f"drop8-{config.method}-points-v1"

    dest_region_dpath = (dest_annot_dpath / "region_models").ensuredir()
    viz_region_dpath = (dest_annot_dpath / "region_models_viz").ensuredir()

    region_model_list = list(region_dpath.glob("*.geojson"))

    if 0:
        # Hack to add special point-only regions
        hacked_regions = ["HK_C001", "HK_C002"]
        for r in hacked_regions:
            region_fpath = empty_region_dpath / (r + ".geojson")
            assert region_fpath.exists()
            region_model_list.append(region_fpath)

    # Handle sites with "xxx" patterns that may be associated with multiple regions
    import kwutil

    region_with_points_patterns = []
    for region_id in regions_with_points:
        if region_id.endswith("xxx"):
            region_id = region_id.replace("xxx", "*")
        region_with_points_patterns.append(region_id)
    region_with_points_pattern = kwutil.util_pattern.MultiPattern.coerce(
        region_with_points_patterns
    )

    status_rows = []
    for region_path in region_model_list:
        region_id = region_path.stem.strip()
        kwcoco_path = (
            kwcoco_bundle_dpath
            / region_id
            / (f"imgonly-{region_id}-rawbands.kwcoco.zip")
        )
        # has_points = region_id in regions_with_points
        has_points = region_with_points_pattern.match(region_id)
        has_kwcoco = kwcoco_path.exists()
        row = {
            "region_id": region_id,
            "annotation_type": region_id.split("_")[1][0],
            "has_points": has_points,
            "has_kwcoco": has_kwcoco,
            "kwcoco_path": kwcoco_path,
        }
        status_rows.append(row)

    import pandas as pd
    import rich

    status_df = pd.DataFrame(status_rows)
    status_df.value_counts("has_kwcoco")
    # rows_to_process = status_df[status_df.has_kwcoco & status_df.has_points]
    # rows_to_process['annotation_type'].value_counts()
    # status_df[~status_df['has_kwcoco']]
    rich.print(status_df)

    polygen_template = ub.codeblock(
        r"""
        python -m geowatch.tasks.poly_from_point.predict \
            --method '{method}' \
            --region_id "{region_id}" \
            --filepath_to_images "{filepath_to_images}" \
            --filepath_to_points "{filepath_to_points}" \
            --filepath_to_region "{filepath_to_region}" \
            --time_prior "{time_prior}" \
            --size_prior "{size_prior}" \
            --filepath_output "{filepath_output}"
        """
    )

    regionviz_template = ub.codeblock(
        """
        geowatch draw_region "{region_fpath}" --fpath "{viz_fpath}"
        """
    )
    force_rerun = 1

    queue = cmd_queue.Queue.create(
        #backend="tmux",
        backend="serial",
        #size=16,
    )
    # queue.add_header_command(
    #     ub.codeblock(
    #         """
    #         pyenv shell 3.10.5
    #         source $(pyenv prefix)/envs/pyenv-geowatch/bin/activate
    #         """
    #     )
    # )
    for row in status_rows:
        region_id = row["region_id"]
        has_kwcoco = row["has_kwcoco"]
        has_points = row["has_points"]
        kwcoco_path = row["kwcoco_path"]

        filepath_output = dest_region_dpath / f"{region_id}.geojson"

        if has_kwcoco and has_points:
            fmtkw = {
                "method": config.method,
                "region_id": region_id,
                "filepath_to_images": None,
                "filepath_to_region": region_path,
                "filepath_output": filepath_output,
                **common_params,
            }
            polygen_cmd = polygen_template.format(**fmtkw)

            if not filepath_output.exists() or force_rerun:
                polygen_job = queue.submit(polygen_cmd, name=f"PolyGen_{region_id}")
            else:
                polygen_job = None

            DO_DRAW = 1
            if DO_DRAW:
                viz_fpath = viz_region_dpath / f"point_viz_{region_id}.png"
                if not viz_fpath.exists() or force_rerun:
                    regionviz_cmd = regionviz_template.format(
                        region_fpath=filepath_output,
                        viz_fpath=viz_fpath,
                    )
                    queue.submit(
                        regionviz_cmd, name=f"Viz_{region_id}", depends=polygen_job
                    )

            final_out_kwcoco_fpath = (
                kwcoco_bundle_dpath
                / region_id
                / (f"pointannv4_ellipse-{region_id}-rawbands.kwcoco.zip")
            )

            if config.ignore_buffer:
                reproject_out_kwcoco_fpath = (
                    kwcoco_bundle_dpath
                    / region_id
                    / (f"_preignore-pointannv4_ellipse-{region_id}-rawbands.kwcoco.zip")
                )
            else:
                reproject_out_kwcoco_fpath = final_out_kwcoco_fpath

            reproject_cmd = ub.codeblock(
                rf"""
                geowatch reproject_annotations \
                    --src "{kwcoco_path}" \
                    --dst "{reproject_out_kwcoco_fpath}" \
                    --region_models="{filepath_output}"
                """
            )
            if not reproject_out_kwcoco_fpath.exists() or force_rerun:
                reproject_job = queue.submit(
                    reproject_cmd, name=f"reproject_{region_id}", depends=polygen_job
                )
            else:
                reproject_job = None

            if config.ignore_buffer:
                ignore_buffer_kwcoco_fpath = final_out_kwcoco_fpath
                add_kwcoco_ignore_buffer_command = ub.codeblock(
                    rf"""
                    python -m geowatch.cli.coco_add_ignore_buffer \
                        --ignore_buffer_size '{config.ignore_buffer}' \
                        --src "{reproject_out_kwcoco_fpath}" \
                        --dst "{ignore_buffer_kwcoco_fpath}"
                    """
                )
                if not ignore_buffer_kwcoco_fpath.exists() or force_rerun:
                    add_ignore_buffer_job = queue.submit(
                        add_kwcoco_ignore_buffer_command,
                        name=f"add_ignore_buffer_{region_id}",
                        depends=reproject_job,
                    )
                else:
                    add_ignore_buffer_job = None  # NOQA

    queue.print_graph()
    queue.print_commands()
    #queue.run()

    print(
        ub.codeblock(
            r"""
        Next Step is to run this splitgen:

        export DST_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
        export DST_BUNDLE_DPATH=$DST_DVC_DATA_DPATH/Drop8-ARA-Median10GSD-V1

        python -m geowatch.cli.queue_cli.prepare_splits \
            --src_kwcocos "$DST_BUNDLE_DPATH"/*/pointannv4_ellipse-*-rawbands.kwcoco.zip \
            --dst_dpath "$DST_BUNDLE_DPATH" \
            --suffix=rawbands_pointannv4_ellipse \
            --backend=tmux --tmux_workers=2 \
            --splits split6 \
            --run=1
        """
        )
    )


if __name__ == "__main__":
    """
    CommandLine:
        python -m geowatch.tasks.poly_from_point.run_all_pointgen
    """
    main()
