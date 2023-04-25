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

apt_ensure(){
    __doc__="
    Checks to see if the packages are installed and installs them if needed.

    The main reason to use this over normal apt install is that it avoids sudo
    if we already have all requested packages.

    Args:
        *ARGS : one or more requested packages

    Example:
        apt_ensure git curl htop

    Ignore:
        REQUESTED_PKGS=(git curl htop)
    "
    # Note the $@ is not actually an array, but we can convert it to one
    # https://linuxize.com/post/bash-functions/#passing-arguments-to-bash-functions
    ARGS=("$@")
    MISS_PKGS=()
    HIT_PKGS=()
    for PKG_NAME in "${ARGS[@]}"
    do
        #apt_ensure_single $EXE_NAME
        RESULT=$(dpkg -l "$PKG_NAME" | grep "^ii *$PKG_NAME")
        if [ "$RESULT" == "" ]; then
            echo "Do not have PKG_NAME='$PKG_NAME'"
            # shellcheck disable=SC2268,SC2206
            MISS_PKGS=(${MISS_PKGS[@]} "$PKG_NAME")
        else
            echo "Already have PKG_NAME='$PKG_NAME'"
            # shellcheck disable=SC2268,SC2206
            HIT_PKGS=(${HIT_PKGS[@]} "$PKG_NAME")
        fi
    done

    if [ "${#MISS_PKGS}" -gt 0 ]; then
        sudo apt install -y "${MISS_PKGS[@]}"
    else
        echo "No missing packages"
    fi
}

###  ENSURE DEPENDENCIES ###

# If on debian/ubuntu ensure the dependencies are installed
if [[ "$(command -v apt)" != "" ]]; then
    apt_ensure ffmpeg tmux jq tree p7zip-full rsync
else
    echo "
    WARNING: Check and install of system packages is currently only supported
    on Debian Linux. You will need to verify that ZLIB, GSL, OpenMP are
    installed before running this script.
    "
fi

# Do everything
python -m pip install setuptools wheel build -U


if [[ "$WATCH_STRICT" == "1" ]]; then
    ./dev/make_strict_req.sh

    python -m pip install -r requirements-strict/gdal.txt

    python -m pip install -r requirements-strict/linting.txt

    # Install the geowatch module in development mode
    python -m pip install -e .[all-strict,headless-strict]

    python -m pip install "dvc[all]>=2.9.3"

    python -m pip install -r requirements-strict/linting.txt

    if ! command -v nvidia-smi &> /dev/null
    then
        echo "nvidia-smi detected"
        python -m pip install -r requirements-strict/mmcv.txt
    else
        echo "nvidia-smi not found, assuming CUDA does not exist"
    fi

else

    python -m pip install -r requirements/gdal.txt

    python -m pip install -r requirements/linting.txt

    python -m pip install "dvc[all]>=2.9.3"

    # Install the geowatch module in development mode
    python -m pip install -e .[all,headless]

    if ! command -v nvidia-smi &> /dev/null
    then
        echo "nvidia-smi detected"
        python -m pip install -r requirements/mmcv.txt
    else
        echo "nvidia-smi not found, assuming CUDA does not exist"
    fi

fi

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
    # NO LONGER NEEDED
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

check_metrics_framework(){
    __doc__="
    Check to see if the IARPA metrics framework is installed
    "
    METRICS_MODPATH=$(python -c "import ubelt; print(ubelt.modname_to_modpath('iarpa_smart_metrics'))")
    if [[ "$METRICS_MODPATH" == "None" ]]; then
        echo "
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        WARNING: IARPA metrics not installed!

        To enable evaluating your results, run this command:

        pip install git+ssh://git@gitlab.kitware.com/smart/metrics-and-test-framework.git -U

        For more information, see:
        https://gitlab.kitware.com/smart/metrics-and-test-framework#installation

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        "
    fi
}
check_metrics_framework


check_gpu_ops_work(){
    # quick check to ensure that GPU operations are generally functional

    xdoctest -m torch --style=google --global-exec "from torch import nn\nimport torch.nn.functional as F\nimport torch" --options="+IGNORE_WHITESPACE"

    python -c "import torch; print(torch.nn.modules.Linear(10, 5).to(0)(torch.rand(10, 10).to(0)).sum().backward())"
}

# Simple tests
set -x
echo "Start simple tests"
EAGER_IMPORT=1 python -c "import watch; print(watch.__version__)"
EAGER_IMPORT=1 python -m watch --help
#EAGER_IMPORT=1 python -m watch hello_world
python -c "import torch; print(torch.cuda.is_available())"
set +x

