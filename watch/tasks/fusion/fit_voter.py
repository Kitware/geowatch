import kwcoco
import ndsampler
import pathlib
from torch.utils import data
import pytorch_lightning as pl

from .methods import voting
from .datasets import onera_2018
from . import utils


def main(args):

    # load dataset
    onera_train = kwcoco.CocoDataset(str(args.train_data_path))
    onera_train_sampler = ndsampler.CocoSampler(onera_train)
    full_train_dataset = onera_2018.OneraDataset(
        onera_train_sampler,
        sample_shape=(1, args.chip_size, args.chip_size),
        channels=args.channels,
    )
    #full_train_dataset.compute_stats(10)

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
    input_dim = train_dataset[0]["images"].shape[1]

    model_var_dict = utils.filter_args(
        vars(args),
        voting.VotingModel.__init__,
    )
    model = voting.VotingModel(
        input_dim=input_dim,
        **model_var_dict
    )

    # trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # fit!
    trainer.fit(model, train_dataloader, valid_dataloader)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path", type=pathlib.Path)

    # dataset / dataloader
    parser.add_argument("--valid_pct", default=0.1, type=float)
    parser.add_argument("--chip_size", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--channels", default=None, type=str)

    # model
    parser = voting.VotingModel.add_model_specific_args(parser)

    # trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    main(args)
