#!/bin/bash
# shellcheck disable=SC2016
__doc__='
Install watch development environment

CommandLine:
    cd $HOME/code/watch
    ./run_developer_setup.sh
'
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
	# Running as a script
	set -eo pipefail
fi


if [[ "$VIRTUAL_ENV" == "" && "$PIP_ROOT_USER_ACTION" != "ignore" ]]; then
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
        RESULT=$(dpkg -l "$PKG_NAME" | grep "^ii *$PKG_NAME" || true)
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

command_exists(){
    __doc__='
    Returns 0 if the command exists and 1 if it does not
    '
    COMMAND=$1
    command -v "$COMMAND" &> /dev/null
}

###  ENSURE DEPENDENCIES ###

# If on debian/ubuntu ensure the dependencies are installed
if [[ "$WITH_APT_ENSURE" != "0" ]]; then
    if command_exists apt; then
        HAS_APT=1
    else
        HAS_APT=0
        echo "
        WARNING: Check and install of system packages is currently only supported
        on Debian Linux. You will need to verify that ZLIB, GSL, OpenMP are
        installed before running this script.
        "
    fi
fi


if [[ "$WITH_MMCV" != "0" ]]; then
    if command_exists nvidia-smi; then
        echo "nvidia-smi detected"
        HAS_NVIDIA_SMI=1
    else
        echo "nvidia-smi not found"
        HAS_NVIDIA_SMI=0
    fi
fi

# User can overwrite this configuration
WATCH_STRICT=${WATCH_STRICT:=0}
WITH_MMCV=${WITH_MMCV:=$HAS_NVIDIA_SMI}
WITH_TENSORFLOW=${WITH_TENSORFLOW:=0}
WITH_DVC=${WITH_DVC:=0}
WITH_AWS=${WITH_AWS:=0}
WITH_COLD=${WITH_COLD:=0}
WITH_MATERIALS=${WITH_MATERIALS:=0}
WITH_APT_ENSURE=${WITH_APT_ENSURE:=$HAS_APT}

echo "

=======================================
____ ____ ____ _ _ _ ____ ___ ____ _  _
| __ |___ |  | | | | |__|  |  |    |__|
|__] |___ |__| |_|_| |  |  |  |___ |  |

=======================================

Environment configuration:

WATCH_STRICT=$WATCH_STRICT
WITH_MMCV=$WITH_MMCV
WITH_DVC=$WITH_DVC
WITH_AWS=$WITH_AWS
WITH_COLD=$WITH_COLD
WITH_MATERIALS=$WITH_MATERIALS
WITH_TENSORFLOW=$WITH_TENSORFLOW
WITH_APT_ENSURE=$WITH_APT_ENSURE
"


# Do everything

if [[ "$WITH_APT_ENSURE" == "1" ]]; then
    apt_ensure ffmpeg tmux jq tree p7zip-full rsync libgsl-dev
fi


if [[ "$WATCH_STRICT" == "1" ]]; then
    ./dev/make_strict_req.sh
    REQUIREMENTS_DPATH=requirements-strict
else
    REQUIREMENTS_DPATH=requirements
fi

# Small python script to compute the extras tag for the pip install
EXTRAS=$(python -c "if 1:
    strict = $WATCH_STRICT
    extras = []
    suffix = '-strict' if strict else ''
    if strict:
        extras.append('runtime' + suffix)
    extras.append('development' + suffix)
    extras.append('tests' + suffix)
    extras.append('optional' + suffix)
    extras.append('headless' + suffix)
    extras.append('linting' + suffix)
    if $WITH_COLD:
        # extras.append('cold' + suffix)
        ...
    if $WITH_MATERIALS:
        extras.append('materials' + suffix)
    if $WITH_DVC:
        extras.append('dvc' + suffix)
    print('[' + ','.join(extras) + ']')
    ")

python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/python_build_tools.txt

# Install the geowatch module in development mode
python -m pip install --prefer-binary -e ".$EXTRAS"

# Post geowatch install requirements

python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/gdal.txt

if [[ "$WITH_COLD" == "1" ]]; then
    # HACK FOR COLD ISSUE
    curl https://data.kitware.com/api/v1/file/6494e95df04fb36854429808/download -o pycold-0.1.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    pip install "astropy==5.2.2"
    #pip install astropy
    #curl https://ipfs.io/ipfs/QmeXUmFML1BBU7jTRdvtaqbFTPBMNL9VGhvwEgrwx2wRew > pycold-311.whl
    #curl ipfs.io/ipfs/QmeXUmFML1BBU7jTRdvtaqbFTPBMNL9VGhvwEgrwx2wRew -o pycold-311.whl
    pip install "pycold-0.1.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    #python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/aws.txt
fi

if [[ "$WITH_AWS" == "1" ]]; then
    python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/aws.txt
fi

if [[ "$WITH_MMCV" == "1" ]]; then
    python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/mmcv.txt
fi

if [[ "$WITH_TENSORFLOW" == "1" ]]; then
    python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/tensorflow.txt
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
            #python -m pip install opencv-python-headless
            python -m pip install -r "$REQUIREMENTS_DPATH"/headless.txt
        fi
    else
        if [[ "$HAS_OPENCV_RETCODE" == "0" ]]; then
            python -m pip uninstall opencv-python -y
        fi
        #python -m pip install opencv-python-headless
        python -m pip install -r "$REQUIREMENTS_DPATH"/headless.txt
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

