I think watch is onto a really nice way of phrasing model training as fit -> predict -> evaluate. 


TODO: netharn-ish plugins for lightning

- [x] callbacks or drawing batches and dumping tensorboard to pngs via matplotlib / seaborn
- [ ] I want to implement netharn style logging (instead of global loggers - because avoiding globals is nice - each instance of the Trainer will get a unique logging object).
- [ ] Directory structure -  I'm making , currently have .  I need to get something to manage the directory structure as well. Probably wherever there is a "version_x" folder I might make a symlink to "recent" for most recent run with that "name" - which is a another thing I'd like control over.


The key idea is if you can phrase a machine learning problem in bash. 
For example, consider this toy problem: 

```bash

# Location of this experiment
DATA_DPATH=$HOME/data/work/toy_change
mkdir -p $DATA_DPATH
cd $DATA_DPATH
# Generate toy datasets
kwcoco toydata vidshapes8-multispectral --bundle_dpath $DATA_DPATH/vidshapes_train
kwcoco toydata vidshapes4-multispectral --bundle_dpath $DATA_DPATH/vidshapes_vali
kwcoco toydata vidshapes2-multispectral --bundle_dpath $DATA_DPATH/vidshapes_test



DATA_DPATH=$HOME/data/work/toy_change
python -m watch.tasks.fusion.fit \
    --train_dataset=$DATA_DPATH/vidshapes_train/data.kwcoco.json \
    --vali_dataset=$DATA_DPATH/vidshapes_vali/data.kwcoco.json \
    --test_dataset=$DATA_DPATH/vidshapes_test/data.kwcoco.json \
    --package_fpath=deployed.pt \
    --max_epochs=1 \
    --max_steps=1 --gpus 1 # [**train_hyperparams]

DATA_DPATH=$HOME/data/work/toy_change
python -m watch.tasks.fusion.predict \
    --test_dataset=$DATA_DPATH/vidshapes_test/data.kwcoco.json \
    --package_fpath=deployed.pt \
    --thresh=0.0605 --gpus 1 \
    --pred_dataset=$DATA_DPATH/vidshapes_test/pred/pred.kwcoco.json  # [**pred_hyperparams]

# jq .images[0] $DATA_DPATH/vidshapes_test/pred/pred.kwcoco.json 
kwcoco show $DATA_DPATH/vidshapes_test/pred/pred.kwcoco.json --gid 1 --channels B1

DATA_DPATH=$HOME/data/work/toy_change
python -m watch.tasks.fusion.evaluate \
    --true_dataset=$DATA_DPATH/vidshapes_test/data.kwcoco.json \
    --pred_dataset=$DATA_DPATH/vidshapes_test/pred/pred.kwcoco.json \
    --eval_dpath=$DATA_DPATH/vidshapes_test/pred/eval  # [**eval_hyperparams]


# tree $DATA_DPATH --filelimit 5 -L 2
# tree $DATA_DPATH/vidshapes_test_pred
```



See Help 

```bash
python -m watch.tasks.fusion.fit --help
python -m watch.tasks.fusion.predict --help
python -m watch.tasks.fusion.evaluate --help
```


### Notes

There are parts of netharn that could be ported to lightning

The logging stuff
    - [x] loss curves (odd they aren't in tensorboard)

The auto directory structure
    - [x] save multiple checkpoints
    - [ ] delete them intelligently

The run management
    - [ ] The netharn/cli/manage_runs.py

The auto-deploy files
    - [x] Use Torch 1.9 Packages instead of Torch-Liberator

Automated dynamics / plugins?

- [X] Rename --dataset argument to --datamodule

- [ ] Rename WatchDataModule to ChangeDataModule

- [ ] Need to figure out how to connect configargparse with ray.tune

- [ ] Distributed Training:
    - [ ] How do do DistributedDataParallel
    - [ ] On one machine
    - [ ] On multiple machines

- [ ] Add Data Modules:
    - [ ] SegmentationDataModule
    - [ ] ClassificationDataModule
    - [ ] DetectionDataModule
    - [ ] <Problem>DataModule
