import kwcoco
import ndsampler
import pathlib
from torch.utils import data
import pytorch_lightning as pl
from torchvision import transforms

from methods import transformer
from datasets import onera_2018
import utils

def main(args):

    transform = utils.Lambda(lambda x: x / 2000.)

    # load dataset
    onera_train = kwcoco.CocoDataset(str(args.train_data_path))
    onera_train_sampler = ndsampler.CocoSampler(onera_train)
    full_train_dataset = onera_2018.OneraDataset(
        onera_train_sampler, 
        sample_shape=(args.time_steps, args.chip_size, args.chip_size),
        channels="<all>",
        transform=transform,
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
    if args.channelwise:
        model_class = transformer.ChannelwiseChangeDetector
    else:
        model_class = transformer.ChangeDetector

    model_var_dict = utils.filter_args(
        vars(args),
        model_class.__init__,
    )
    model = model_class(
        **model_var_dict
    )
    
    # trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # prime the model, it has a lazy linear layer
    batch = next(iter(train_dataloader))
    result = model(batch["images"][[0],...])
    
    # fit!
    trainer.fit(model, train_dataloader, valid_dataloader)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path", type=pathlib.Path)
    
    # dataset / dataloader
    parser.add_argument("--channelwise", action="store_true")
    parser.add_argument("--valid_pct", default=0.1, type=float)
    parser.add_argument("--chip_size", default=128, type=int)
    parser.add_argument("--time_steps", default=2, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)

    temp_args, _ = parser.parse_known_args()
    
    # model
    if temp_args.channelwise:
        model_class = transformer.ChannelwiseChangeDetector
    else:
        model_class = transformer.ChangeDetector
    parser = model_class.add_model_specific_args(parser)
    
    # trainer
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    main(args)
    
