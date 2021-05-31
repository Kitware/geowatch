import kwcoco
import ndsampler
import pathlib
from torch.utils import data
import pytorch_lightning as pl
import tifffile
import numpy as np
import tempfile

from methods import baseline
from datasets import onera_2018
import onera_experiment_train as onera_experiment
import utils

fname_template = "{location}/{bands}-{frame_no}.tif"

def main(args):
    
    onera_test = kwcoco.CocoDataset(str(args.test_data_path))
    onera_test_sampler = ndsampler.CocoSampler(onera_test)

    predict_dataset = onera_2018.OneraDataset(
        onera_test_sampler, 
        sample_shape=(2, None, None),
        channels=onera_experiment.channel_combos[args.channel_set],
        mode="predict",
    )
    predict_dataloader = data.DataLoader(predict_dataset, batch_size=1)

    model = baseline.ChangeDetector.load_from_checkpoint(args.model_checkpoint_path)
    model.eval(); model.freeze();

    tmp_root = tempfile.TemporaryDirectory()
    trainer = pl.Trainer(default_root_dir=tmp_root.name)

    results = trainer.predict(model, predict_dataloader)

    for video, result_stack in zip(onera_test.dataset["videos"], results):

        frames = [
            img
            for img in onera_test.imgs.values()
            if img["video_id"] == video["id"]
        ][1:]

        for frame, result in zip(frames, result_stack[0].detach().cpu().numpy()):

            height, width = result.shape[0], result.shape[1]

            result_fname = args.results_dir / fname_template.format(
                location=video["name"], 
                bands=args.channel_set,
                frame_no=frame["frame_index"],
            )
            result_fname.parents[0].mkdir(exist_ok=True)
            tifffile.imwrite(result_fname, result)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test_data_path", type=pathlib.Path)
    parser.add_argument("channel_set")
    parser.add_argument("model_checkpoint_path", type=pathlib.Path)
    parser.add_argument("results_dir", type=pathlib.Path)
    args = parser.parse_args()
    
    main(args)