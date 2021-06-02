import kwcoco
import ndsampler
import pathlib
from torch.utils import data
import pytorch_lightning as pl
import tifffile
import numpy as np
import tqdm

from methods import voting
from datasets import onera_2018
import onera_experiment_train as onera_experiment
import utils

fname_template = "{location}/{bands}-{frame_no}.tif"

def main(args):
    
    onera_test = kwcoco.CocoDataset(str(args.test_data_path))
    onera_test_sampler = ndsampler.CocoSampler(onera_test)

    predict_dataset = onera_2018.SimpleDataset(
        onera_test_sampler, 
        sample_shape=(1, None, None),
        channels=args.channels,
        mode="test",
    )
    predict_dataloader = data.DataLoader(predict_dataset, batch_size=1)

    model = voting.VotingModel.load_from_checkpoint(args.model_checkpoint_path)
    model.eval(); model.freeze();

    results = [
        model(example["images"][None])[0]
        for example in tqdm.tqdm(predict_dataset)
    ]

    for video, result_stack in zip(onera_test.dataset["videos"], results):

        frames = [
            img
            for img in onera_test.imgs.values()
            if img["video_id"] == video["id"]
        ]

        result_stack = result_stack.detach().cpu().numpy()
        for frame, result in zip(frames, result_stack):

            result_fname = args.results_dir / fname_template.format(
                location=video["name"], 
                bands="voting_"+args.channels.replace("|", "-"),
                frame_no=frame["frame_index"],
            )
            result_fname.parents[0].mkdir(parents=True, exist_ok=True)

            tifffile.imwrite(result_fname, result)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test_data_path", type=pathlib.Path)
    parser.add_argument("channels")
    parser.add_argument("model_checkpoint_path", type=pathlib.Path)
    parser.add_argument("results_dir", type=pathlib.Path)
    args = parser.parse_args()
    
    main(args)
