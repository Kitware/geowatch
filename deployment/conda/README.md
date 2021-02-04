# Setup WATCH Development Environment

This document describes how to setup a development environment for the SMART WATCH project using Conda. Please follow the instructions step-by-step.

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

### Create SMART Conda Environment

The instructions below can be used to create the SMART Conda environment.

```bash
mkdir SMART_DIR (pick a name of your choice)
cd SMART_DIR
git clone https://gitlab.kitware.com/smart/watch.git
cd watch
conda env create -f deployment/conda/conda_env.yml
source activate watch
```

To deactivate the watch environment, run:

```bash
source deactivate
```
To remove the watch environment, run:

```bash
source deactivate (if watch is activated before)
conda remove --name watch --all
```
To update the watch environment when new packages have been added, run:

```bash
source activate watch
conda env update -f deployment/conda/conda_env.yml
```
