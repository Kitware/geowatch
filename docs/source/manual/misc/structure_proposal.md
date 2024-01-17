# Directory Structure for Multi-task Integration


## Design Goals (2021): 

    - [ ] Standard format in which individual tasks (e.g. semantic segmentation, fusion, etc...) can be executed.

    - [ ] Make common code shared among teams for increased re-usability.

    - [ ] All tasks can be executed through a common fit and predict modules with similar APIs.


## Design Goals (2023): 

    - [ ] Have tools infer sensible defaults whenever possible.

    - [ ] 


## Directory Structure:

    watch
    ├── datasets                                    # generatic re-usable dataset templates
    │   ├── video.py            
    │   ├── segmentation.py      
    │   ├── detection.py         
    │   ├── classification.py    
    │   └── ...                                
    ├── tasks                                       # Individual tasks for each group
    │   ├── <fusion_task> 
    │   │   └── ...        
    │   ├── <rutgers_task>    
    │   │   ├── <rutgers_task>_dataset.py           # method specific datasets
    │   │   ├── models                              # method specific models
    │   │   │   ├── <rutgers_task>_model.py
    │   │   ├── utils                               # method specific utilities
    │   │   │   ├── <rutgers_task>_specific_util.py
    │   │   ├── fit.py
    │   │   ├── predict.py
    │   │   └── ...        
    │   ├── <u_maryland_task>      
    │   │   └── ...        
    │   ├── <u_conn_task>         
    │   │   └── ...        
    │   ├── <u_kentucky_task>
    │   │   └── ...        
    │   └── ...        
    ├── models                                      # generaic or common models. This can also contain all models.
    │   ├── resnet.py            
    │   ├── unet.py      
    │   ├── segnet.py         
    │   ├── deeplabv3.py   
    │   ├── aspp.py
    │   ├── swin.py
    │   └── ... 
    ├── utils                                       # generic re-usable utilities
    │   ├── util_raster.py      
    │   ├── util_girder.py         
    │   ├── util_visualization.py    
    │   └── ...    
    ├── gis                                         # 
    ├── demo                                        # Demos
    ├── validation                                  # 
    └── ...

The name of each task, can be chosen by the owners of that task.
Code in each task folder should contain at minimum two scripts: 

* `fit.py`  for training a model from a specified dataset, and
* `predict.py` for predicting on a specified dataset given a specified model.

Other code in each task folder can be arbitrary, and task-developers should be
able to expect that their task folder to be somewhat sandboxed, and
other developers will not introduce conflicts.

### Usage:
This structure allows for training and evaluating tasks independently, 
or evaluate models jointly (meaning concatenated features) through 
the fusion module. When a method stores the best performing model, 
it saves a "deployed.zip" zipfile which contains dataset hyperparameters, 
model weights, and method specific configurations. 

### Fit API:
To train a model, we expect that the task-specific fit script will use 
a command line interface somewhat like this:

```bash
python -m geowatch.tasks.<task_name>.fit --train_dataset=<path-to-kwcoco> --vali_dataset=<path-to-kwcoco> <additional hyperparam config>
```

The `<additional hyperparam config>` could be additional command line 
parameters (e.g. `--lr=3e-4`, `--batch_size=4`) or a path to a config 
file (e.g. `--config=<path-to-yml>`), which might also contain the 
train and validation dataset paths (although we strongly recommend 
that passing paths to the training / validation / testing datasets 
be specifiable via the command line). Note the best-of-both-worlds 
can be obtained by using `scriptconfig` (https://pypi.org/project/scriptconfig/) 
for configuration management.

At a minimum, this fit task should produce a trained state-dict for a 
particular model. Ideally the task will use torch-liberator 
(https://pypi.org/project/torch-liberator/) to package that state-dict 
with the model code itself, and a json file containing relevant metadata 
into a standalone deploy zipfile.

NOTE: if your method does not require learning parameters of a model, it is
fine to omit the "fit" script and just provide "predict".


### Predict API:
To predict with a model, we expect that there will be a task-specific 
predict script. This should take model weights to predict with, 
and a kwcoco dataset to predict on.

```bash
python -m geowatch.tasks.<task_name>.predict --deployed=<path-to-deploy-zipfile> --dataset=<path-to-kwcoco> <additional prediction config>
```

The output of this script should be a modified version of the input 
kwcoco file with additional annotations / auxiliary channels predictions. 
Again `<additional prediction config>` can be a config file, or additional 
command line arguments (again we suggest using `scriptconfig`).


### Evaluation:

Evaluation will be handled by using the predict API in conjunction with an
external evaluation tool.


## Examples:


Example invocations of fit and predict scripts may look like this:


```bash
    python -m geowatch.tasks.rutgers.fit --train_dataset=drop0-train.kwcoco.json --config=train_config_v1.yml

    python -m geowatch.tasks.rutgers.predict --deployed=model_v1.zip --dataset=drop0-test.kwcoco.json

    python -m geowatch.tasks.invariants.fit --train_dataset=drop0-train.kwcoco.json --vali_dataset=drop0-train.kwcoco.json --model=custom_arch_v1 --init=<path/to/pretrained/state.pt> --lr=1e-3 --workers=8 --workdir=$HOME/work/smart --name=myexpt_v1
    python -m geowatch.tasks.invariants.predict --deployed=$HOME/work/smart/myexpt_v1/deployed.zip --dataset=drop0-test.kwcoco.json --output=drop0-test-predictions.kwcoco.json

    python -m geowatch.tasks.fusion.fit --config fusion_fit_config_v5.yml
    python -m geowatch.tasks.fusion.predict --config fusion_predict_config_v5.yml
```
