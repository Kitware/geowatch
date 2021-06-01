import fit
import pathlib
import pytorch_lightning as pl

from methods import baseline
from datasets import onera_2018
import utils


channel_combos = {
    "all": "<all>",
    "uv": "B01",
    "bgr": "B02|B03|B04",
    "vnir": "B05|B06|B07|B08|B8A",
    "swir": "B09|B10|B11|B12",
}

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path", type=pathlib.Path)
    
    # dataset / dataloader
    parser.add_argument("--valid_pct", default=0.1, type=float)
    parser.add_argument("--chip_size", default=128, type=int)
    parser.add_argument("--time_steps", default=2, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
#     parser.add_argument("--channels", default=None, type=str)
    
    # model
    parser = baseline.ChangeDetector.add_model_specific_args(parser)
    
    # trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    
    for key, channels in channel_combos.items():
        args.channels = channels
        args.default_root_dir = f"experiments/onera/{key}"
        fit.main(args)