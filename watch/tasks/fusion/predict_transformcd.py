import kwcoco
import ndsampler
import pathlib
from torch.utils import data
import pytorch_lightning as pl
import tifffile
import numpy as np
import tqdm

from methods import transformer
from datasets import onera_2018
import utils

fname_template = "{location}/{bands}-{frame_no}.tif"

def main(args):
    
    onera_test = kwcoco.CocoDataset(str(args.test_data_path))
    onera_test_sampler = ndsampler.CocoSampler(onera_test)

    predict_dataset = onera_2018.OneraDataset(
        onera_test_sampler, 
        sample_shape=(2, args.window_size, args.window_size),
        window_overlap=(0, args.window_overlap, args.window_overlap)
        channels="<all>",
        mode="test",
        transform=utils.Lambda(lambda x: x/2000.)
    )

    model = baseline.ChangeDetector.load_from_checkpoint(args.model_checkpoint_path)
    model.eval(); model.freeze();
    
    result_canvases = {
        video["id"]: np.full([video["height"], video["width"]], -2)
        for video in onera_test.dataset["videos"]
    }
    result_counts = {
        video["id"]: np.full([video["height"], video["width"]], 0)
        for video in onera_test.dataset["videos"]
    }
    target_canvases = {
        video["id"]: np.full([video["height"], video["width"]], -2)
        for video in onera_test.dataset["videos"]
    }
    
    for example, meta in zip(tqdm.tqdm(train_dataset), train_dataset.sample_grid):
        images, labels = example["images"], example["labels"]
        changes = labels[1:] != labels[:-1]
        
        preds = model(images[None])[0]
        
        result_canvases[meta["vidid"]][meta["space_slice"]] += preds
        result_counts[meta["vidid"]][meta["space_slice"]] += 1
        target_canvases[meta["vidid"]][meta["space_slice"]] = changes
        
    results = {
        key: canvas / result_counts[key]
        for key, canvas in result_canvases.items()
    }
    targets = target_canvases
    

    for video in onera_test.dataset["videos"]:
        
        result_stack = results[video["id"]]
        target_stack = targets[video["id"]]

        frames = [
            img
            for img in onera_test.imgs.values()
            if img["video_id"] == video["id"]
        ][1:]

        result_stack = result_stack.detach().cpu().numpy()
        target_stack = target_stack.detach().cpu().numpy()
        for frame, result, target in zip(frames, result_stack, target_stack):

            result_fname = args.results_dir / fname_template.format(
                location=video["name"], 
                bands=args.channel_set,
                frame_no=frame["frame_index"],
            )
            target_fname = args.results_dir / fname_template.format(
                location=video["name"], 
                bands="target",
                frame_no=frame["frame_index"],
            )
            result_fname.parents[0].mkdir(parents=True, exist_ok=True)

            tifffile.imwrite(result_fname, result)
            tifffile.imwrite(target_fname, target)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test_data_path", type=pathlib.Path)
    parser.add_argument("model_checkpoint_path", type=pathlib.Path)
    parser.add_argument("results_dir", type=pathlib.Path)
    parser.add_argument("--window_size", default=128, type=int)
    parser.add_argument("--window_overlap", default=0.1, type=float)
    args = parser.parse_args()
    
    main(args)
