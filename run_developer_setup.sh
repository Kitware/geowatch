#!/bin/bash
__doc__='
Install watch development environment

CommandLine:
    cd $HOME/code/watch
    ./run_developer_setup.sh
'


if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "NOT INSIDE OF A VIRTUALENV. This script may not run correctly"
fi 

#pip install -r requirements/no-deps.txt

# Do everything
python -m pip install pip setuptools build -U

pip install scikit-image  # not sure why this doesn't grab latest in req.txt (look into that)

python -m pip install -r requirements.txt -v

# Install the watch module in development mode
python -m pip install -e .

# Install more fragile dependencies
# pip install imgaug>=0.4.0
# pip install netharn>=0.5.16
#pip install GDAL>=3.5.0 --find-links https://girder.github.io/large_image_wheels -U

# TODO: we should skip trying to install gdal if possible

python -m pip install GDAL==3.4.1 --find-links https://girder.github.io/large_image_wheels -U

python -m pip install dvc[all]>=2.9.3

python -m pip install lru-dict || echo "unable to install lru-dict"

fix_opencv_conflicts(){
    __doc__="
    Check to see if the wrong opencv is installed, and perform steps to clean
    up the incorrect libraries and install the desired (headless) ones.
    "
    # Fix opencv issues
    python -m pip freeze | grep "opencv-python=="
    HAS_OPENCV_RETCODE="$?"
    python -m pip freeze | grep "opencv-python-headless=="
    HAS_OPENCV_HEADLESS_RETCODE="$?"

    # VAR == 0 means we have it
    if [[ "$HAS_OPENCV_HEADLESS_RETCODE" == "0" ]]; then
        if [[ "$HAS_OPENCV_RETCODE" == "0" ]]; then
            python -m pip uninstall opencv-python opencv-python-headless -y
            python -m pip install opencv-python-headless
        fi
    else
        if [[ "$HAS_OPENCV_RETCODE" == "0" ]]; then
            python -m pip uninstall opencv-python -y
        fi
        python -m pip install opencv-python-headless
    fi
}

torch_on_3090(){
    # https://github.com/pytorch/pytorch/issues/31285
    # Seems like we need to work from source:
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    # if you are updating an existing checkout
    git submodule sync
    git submodule update --init --recursive --jobs 0
    python -m pip install . -v
}

fix_opencv_conflicts

# Simple tests
set -x
echo "Start simple tests"
EAGER_IMPORT=1 python -c "import watch; print(watch.__version__)"
EAGER_IMPORT=1 python -m watch --help
EAGER_IMPORT=1 python -m watch hello_world
python -c "import torch; print(torch.cuda.is_available())"
set +x
