def main():
    import cmd_queue
    import ubelt as ub

    queue = cmd_queue.Queue.create(backend="tmux", size=4)
    queue.add_header_command(
        ub.codeblock(
            """
            pyenv shell 3.10.5
            source $(pyenv prefix)/envs/pyenv-geowatch/bin/activate
            """
        )
    )
    template = ub.codeblock(
        r"""
        python -m geowatch.tasks.poly_from_point.predict \
        --method 'sam' \
        --filepath_to_images "{file_path_to_images}" \
        --filepath_to_region "{file_path_to_region}" \
        """
    )
    region_model_path = ub.Path(
        "/mnt/ssd2/data/dvc-repos/smart_phase3_data/annotations/drop8/region_models/"
    )
    region_model_list = list(region_model_path.glob("*.geojson"))
    kwcoco_dpath = ub.Path(
        "/mnt/ssd2/data/dvc-repos/smart_phase3_data/Aligned-Drop8-ARA/"
    )
    for region_path in region_model_list:
        region_id = region_path.stem
        kwcoco_path = (
            kwcoco_dpath / region_id / (f"imgonly-{region_id}-rawbands.kwcoco.zip")
        )
        if kwcoco_path.exists():
            fmtkw = {
                "file_path_to_images": kwcoco_path,
                "file_path_to_region": region_path,
            }
            cmd = template.format(**fmtkw)
            queue.submit(cmd, name=f"From point: {region_id} ")
    queue.print_commands()
    queue.run()


if __name__ == "__main__":
    main()
