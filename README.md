# WATCH

[![master-pipeline](https://gitlab.kitware.com/smart/watch/badges/master/pipeline.svg)](https://gitlab.kitware.com/smart/watch/badges/master/pipeline.svg)
[![master-coverage](https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg)](https://gitlab.kitware.com/smart/watch/badges/master/coverage.svg)


This repository addresses the algorithmic challenges of the IARPA SMART program. The goal of this software is analyze space-based imagery to perform broad-area search for natural and anthropogenic events and characterize their extent and progression in time and space.

## Getting Started

### Install Conda

Conda3 is required to set up the environment using Python 3. Follow the instructions below to install miniconda3 for Linux. For Windows 10 users, the Windows Subsystem for Linux (WSL) allows you to run Linux within Windows.

```bash
# Download the conda install script into a temporary directory
mkdir -p ~/tmp
cd ~/tmp
CONDA_INSTALL_SCRIPT=Miniconda3-latest-Linux-x86_64.sh
curl https://repo.anaconda.com/miniconda/$CONDA_INSTALL_SCRIPT > $CONDA_INSTALL_SCRIPT
chmod +x $CONDA_INSTALL_SCRIPT

# Install miniconda to user local directory
_CONDA_ROOT=$HOME/.local/conda
sh $CONDA_INSTALL_SCRIPT -b -p $_CONDA_ROOT
# Activate the basic conda environment
source $_CONDA_ROOT/etc/profile.d/conda.sh
# Update the base
conda update --name base conda --yes
```

### Create SMART Conda environment

The instructions below can be used to create the SMART Conda environment.

```bash
mkdir SMART_DIR #(pick a name of your choice)
cd SMART_DIR
git clone https://gitlab.kitware.com/smart/watch.git
cd watch
conda env create -f deployment/conda/conda_env.yml
conda activate watch
```

To deactivate the watch environment, run:

```bash
conda deactivate
```
To remove the watch environment, run:

```bash
conda deactivate #(if watch is activated before)
conda remove --name watch --all
```
To update the watch environment when new packages have been added, run:

```bash
conda activate watch
conda env update -f deployment/conda/conda_env.yml
```

### Installation

The WATCH Python module can then be installed with `pip` via the following command, where `/path/to/watch` is the absolute path to the directory containing this README.md file.

```
pip install -e /path/to/watch
```

### Running the Algorithm Toolkit (ATK) example project

Ensure that you have already setup the SMART Conda enviornment.

Then enter the following commands in your terminal to run the ATK example project:

```bash
cd atk/example
alg run
```

Point your browser to http://localhost:5000/. You should see the development environment welcome page.

Refer to the [development enviornment](https://algorithm-toolkit.readthedocs.io/en/latest/dev-environment.html#) portion of the
[atk docs](https://algorithm-toolkit.readthedocs.io/en/latest/index.html) for a crash course on how to use the web-based development environment.


## Running tests

We're using the `pytest` module for running unit tests.  Unit tests should be added into the `tests` directory and files should be prefixed with `test_`.

The `run_tests.py` script provided here will run all tests in the `tests` directory.

## Adding submodules

Library code can be added to the relevant subdirectory under the `watch` directory.  The current submodules are as follows:

- datacube/atmosphere
- datacube/cloud
- datacube/registration
- datacube/reflectance
- features/materials
- features/semantics
- features/invariants
- features/reflectance
- fusion
- sequencing
- validation
- tools
- utils

## Adding command line tools

New Python command line scripts can be added under the `watch/tools` directory.  To have the command line tool be installed with the module, an entry can be added to the `setup.py` setup call, under `entrypoints['console_scripts']`.

Scripts that don't quite belong in the WATCH Python module itself (e.g. due to a lack of general purpose use, or lack of polish) can be added to the `scripts` directory.
