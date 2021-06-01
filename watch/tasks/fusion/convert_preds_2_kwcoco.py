import kwcoco
import kwimage
import tifffile
import pathlib
import utils
    
def main(args):

    base_ds = kwcoco.CocoDataset(str(args.base_kwcoco_path))

    results_ds = kwcoco.CocoDataset()
    results_ds.add_category("change")

    for video in base_ds.dataset["videos"]:
        results_ds.add_video(
            name=video["name"],
            id=video["id"],
            width=video["width"],
            height=video["height"],
            target_gsd=video["target_gsd"],
            warp_wld_to_vid=video["warp_wld_to_vid"],
            min_gsd=video["min_gsd"],
            max_gsd=video["max_gsd"],
        )

        frames = [
            img
            for img in base_ds.imgs.values()
            if img["video_id"] == video["id"]
        ][1:]

        video_dir = args.predictions_dir / video["name"]

        for frame in frames:
            results_ds.add_image(
                id=frame["id"],
                name="{}-{}".format(video["name"], frame["frame_index"]),
                width=frame["width"],
                height=frame["height"],
                video_id=video["id"],
                frame_index=frame["frame_index"],
                img_to_vid=frame["img_to_vid"], 
                warp_img_to_vid=frame["warp_img_to_vid"],
            )
            
            glob_pattern = f"*-{frame['frame_index']}.tif"
            band_paths = sorted(list(video_dir.glob(glob_pattern)))
            target_idx = [
                path.stem.split("-")[0]
                for path in band_paths
            ].index("target")
            target_path = band_paths.pop(target_idx)
            
            segmentation = tifffile.imread(target_path)
            segmentation = kwimage.Mask(segmentation, "f_mask").to_coco()

            results_ds.add_annotation(
                frame["id"],
                category_id=1,
                bbox=[0, 0, video["width"], video["height"]],
                segmentation=segmentation,
            )

            for band_path in band_paths:
                utils.add_auxiliary(
                    results_ds,
                    frame["id"],
                    str(band_path.absolute()), 
                    band_path.stem.split("-")[0], 
                    aux_width=video["width"],
                    aux_height=video["height"],
                    warp_aux_to_img=None)

    results_ds.dump(str(args.kwcoco_dest))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("base_kwcoco_path", type=pathlib.Path)
    parser.add_argument("predictions_dir", type=pathlib.Path)
    parser.add_argument("kwcoco_dest", type=pathlib.Path)
    args = parser.parse_args()
    
    main(args)