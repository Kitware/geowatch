import pytorch_lightning as pl

import datasets
import methods
import utils

def main(args):
    
    # init method from args
    method_class = getattr(methods, args.method)
    method_var_dict = utils.filter_args(
        vars(args),
        method_class.__init__,
    )
    method = method_class(
        **method_var_dict
    )
    
    # init dataset from args
    dataset_class = getattr(datasets, args.dataset)
    dataset_var_dict = utils.filter_args(
        vars(args),
        dataset_class.__init__,
    )
    dataset_var_dict["preprocessing_step"] = method.preprocessing_step
    dataset = dataset_class(
        **dataset_var_dict
    )
    dataset.setup("fit")
    
    # init trainer from args
    trainer = pl.Trainer.from_argparse_args(args)

    # prime the model, incase it has a lazy layer
    batch = next(iter(dataset.train_dataloader()))
    result = method(batch["images"][[0],...])
    
    # fit the model
    trainer.fit(method, dataset)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("method")
    
    # parse the dataset and method strings
    temp_args, _ = parser.parse_known_args()
    
    # get the dataset and method classes
    dataset_class = getattr(datasets, temp_args.dataset)
    method_class = getattr(methods, temp_args.method)
    
    # add the appropriate args to the parse 
    # for dataset, method, and trainer
    parser = dataset_class.add_data_specific_args(parser)
    parser = method_class.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    
    # parse and pass to main
    args = parser.parse_args()
    main(args)
