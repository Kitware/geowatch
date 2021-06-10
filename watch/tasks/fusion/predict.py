# import pytorch_lightning as pl
import pathlib
import numpy as np
import tqdm
import tifffile

import datasets
import methods
import utils

fname_template = "{location}/{bands}-{frame_no}.tif"

def main(args):
    
    # init dataset from args
    dataset_class = getattr(datasets, args.dataset)
    dataset_var_dict = utils.filter_args(
        vars(args),
        dataset_class.__init__,
    )
    dataset = dataset_class(
        **dataset_var_dict
    )
    dataset.setup("test")
    
    test_dataset = dataset.test_dataset
    test_dataloader = dataset.test_dataloader()
    
    # init method from checkpoint
    method_class = getattr(methods, args.method)
    method = method_class.load_from_checkpoint(args.checkpoint_path)
    method.eval(); method.freeze();
    
    # set up canvases for prediction
    result_canvases = {
        video["id"]: np.full([
                video["num_frames"]-1, video["height"], video["width"],
            ], -2)
        for video in test_dataset.dataset["videos"]
    }
    result_counts = {
        video["id"]: np.full([
                video["num_frames"]-1, video["height"], video["width"],
            ], 0)
        for video in test_dataset.dataset["videos"]
    }
    target_canvases = {
        video["id"]: np.full([
                video["num_frames"]-1, video["height"], video["width"],
            ], -2)
        for video in test_dataset.dataset["videos"]
    }
    
    # fill canvases
    for example, meta in zip(tqdm.tqdm(test_dataloader), test_dataset.sample_grid):
        images, labels = example["images"], example["labels"]
        changes = labels[0, 1:] != labels[0, :-1]
        
        preds = model(images)[0]
        
        space_time_slice = (meta["time_slice"],) + meta["space_slice"]
        
        result_canvases[meta["vidid"]][space_time_slice] += preds
        result_counts[meta["vidid"]][space_time_slice] += 1
        target_canvases[meta["vidid"]][space_time_slice] = changes
        
    results = {
        key: canvas / result_counts[key]
        for key, canvas in result_canvases.items()
    }
    targets = target_canvases
    
    # save canvases to disk
    for video in test_dataset.dataset["videos"]:
        
        result_stack = results[video["id"]]
        target_stack = targets[video["id"]]

        frames = [
            img
            for img in test_dataset.imgs.values()
            if img["video_id"] == video["id"]
        ][1:]

        result_stack = result_stack.detach().cpu().numpy()
        target_stack = target_stack.detach().cpu().numpy()
        for frame, result, target in zip(frames, result_stack, target_stack):

            result_fname = args.results_dir / fname_template.format(
                location=video["name"], 
                bands=args.tag,
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

if __name == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("method")
    parser.add_argument("tag")
    parser.add_argument("checkpoint_path", type=pathlib.Path)
    
    # parse the dataset and method strings
    temp_args, _ = parser.parse_known_args()
    
    # get the dataset and method classes
    dataset_class = getattr(datasets, temp_args.dataset)
    
    # add the appropriate args to the parse 
    # for dataset, method, and trainer
    parser = dataset_class.add_data_specific_args(parser)
    
    # parse and pass to main
    args = parser.parse_args()
    
    assert args.batch_size == 1
    
    main(args)