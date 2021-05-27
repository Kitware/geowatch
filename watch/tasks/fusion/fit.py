import kwcoco
import ndsampler
import pathlib
from torch.utils import data
import pytorch_lightning as pl

from methods import baseline
from datasets import onera_2018
import utils

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
    
    # model
    parser = baseline.ChangeDetector.add_model_specific_args(parser)
    
    # trainer
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    
    # load dataset
    onera_train = kwcoco.CocoDataset(str(args.train_data_path))
    onera_train_sampler = ndsampler.CocoSampler(onera_train)
    full_train_dataset = onera_2018.OneraDataset(
        onera_train_sampler, 
        (args.time_steps, args.chip_size, args.chip_size),
    )
    
    # split into train/valid
    num_examples = len(full_train_dataset)
    num_valid = int(args.valid_pct * num_examples)
    num_train = num_examples - num_valid
    
    train_dataset, valid_dataset = data.random_split(
        full_train_dataset, 
        [num_train, num_valid],
    )
    
    # dataloaders
    train_dataloader = data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    valid_dataloader = data.DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    
    # model
    input_dim = len(list(onera_train.imgs.values())[0]["auxiliary"]) # TODO: maybe theres a better way to determine this...
    model_var_dict = utils.filter_args(
        vars(args),
        baseline.ChangeDetector.__init__,
    )
    model = baseline.ChangeDetector(
        input_dim=input_dim,
#         feature_dim=args.feature_dim,
#         lr=args.learning_rate,
        **model_var_dict
    )
    
    # trainer
    trainer = pl.Trainer.from_argparse_args(args)
    
    # fit!
    trainer.fit(model, train_dataloader, valid_dataloader)