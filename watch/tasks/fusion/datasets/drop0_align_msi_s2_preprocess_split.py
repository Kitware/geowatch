"""
TODO: this should be moved to a scripts directory
"""
import kwcoco
import itertools as it
import pathlib
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("orig_kwcoco_path", type=pathlib.Path)
    args = parser.parse_args()

    project_ds = kwcoco.CocoDataset(str(args.orig_kwcoco_path.expanduser()))

    valid_sites = [
        'N33.116391W081.790652_N33.161971W081.750707',
    ]
    train_sites = [
        'N23.932504E052.180122_N24.002794E052.286242',
        'N37.655717E128.651836_N37.668362E128.687321',
    ]

    project_ds.remove_images([
        image
        for image in project_ds.dataset["images"]
        if "S2-TrueColor" not in image["sensor_candidates"]
    ])
    project_ds.remove_videos([2, 3])

    for image in project_ds.dataset["images"]:
        image["img_to_vid"] = {
            'type': 'affine',
            'matrix': [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]}
        image["warp_img_to_vid"] = {
            'type': 'affine',
            'matrix': [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]}

        for band in image["auxiliary"]:
            band["warp_aux_to_img"]["type"] = "affine"

    for video in project_ds.dataset["videos"]:
        video_id = video["id"]

        video["width"] = max([
                image["width"]
                for image in project_ds.dataset["images"]
                if image["video_id"] == video["id"]
            ])
        video["height"] = max([
                image["height"]
                for image in project_ds.dataset["images"]
                if image["video_id"] == video["id"]
            ])
        video["num_frames"] = len([
                image
                for image in project_ds.dataset["images"]
                if image["video_id"] == video["id"]
            ])
        video["available_channels"] = list(set(it.chain.from_iterable([
                [
                    band["channels"]
                    for band in image["auxiliary"]
                ]
                for image in project_ds.dataset["images"]
                if image["video_id"] == video_id
            ])))
        video["warp_wld_to_vid"] = {
            'type': 'affine',
            'matrix': [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]}
        video["target_gsd"] = 10.0
        video["min_gsd"] = 10.0
        video["max_gsd"] = 10.0

    project_ds.validate()

    print("Available names:", [
            video["name"]
            for video in project_ds.dataset["videos"]
        ])

    train_ds = project_ds.copy()
    valid_ds = project_ds.copy()

    combo_path = args.orig_kwcoco_path.parent / ("combo_" + args.orig_kwcoco_path.name)
    project_ds.dump(str(combo_path.expanduser()))

    train_ds.remove_videos(valid_sites)
    valid_ds.remove_videos(train_sites)

    train_path = args.orig_kwcoco_path.parent / ("train_" + args.orig_kwcoco_path.name)
    train_ds.dump(str(train_path.expanduser()))

    valid_path = args.orig_kwcoco_path.parent / ("valid_" + args.orig_kwcoco_path.name)
    valid_ds.dump(str(valid_path.expanduser()))


if __name__ == '__main__':
    main()
