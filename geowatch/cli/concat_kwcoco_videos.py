import argparse
import sys

import ubelt
import kwcoco

from geowatch.utils import kwcoco_extensions


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate KWCOCO datasets at the video level")

    parser.add_argument("--dst",
                        type=str,
                        help="Path to output KWCOCO dataset")
    parser.add_argument('src_kwcoco_datasets',
                        type=str,
                        nargs='+',
                        help="Paths to input KWCOCO datasets to concatenate")

    concat_kwcoco_datasets(**vars(parser.parse_args()))


def concat_kwcoco_datasets(src_kwcoco_datasets, dst):
    first, *rest = src_kwcoco_datasets

    out_dset = kwcoco.CocoDataset(first).copy()

    def _build_video_lookup(kwcoco_dset):
        video_metas = kwcoco_dset.videos().lookup(('id', 'name'))
        return dict(zip(video_metas['name'], video_metas['id']))

    out_dset_videos_lookup = _build_video_lookup(out_dset)

    for dset_path in rest:
        next_id = max(ubelt.flatten(out_dset.videos().images.lookup('id')))
        dset = kwcoco.CocoDataset(dset_path)

        dset_videos_lookup = _build_video_lookup(dset)

        for video_name, video_id in dset_videos_lookup.items():
            if video_name in out_dset_videos_lookup:
                out_video_id = out_dset_videos_lookup[video_name]

                for image_group in dset.videos([video_id]).images:
                    for image in image_group.objs:
                        next_id += 1
                        image['id'] = next_id
                        image['video_id'] = out_video_id

                        out_dset.add_image(**image)
            else:
                out_dset.add_video(dset.videos([video_id]))

    # Ensure video frames are in the right order
    kwcoco_extensions.reorder_video_frames(out_dset)

    out_dset.dump(dst, indent=2, newlines=True)


if __name__ == "__main__":
    sys.exit(main())
