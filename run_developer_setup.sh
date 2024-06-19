#!/bin/bash
# shellcheck disable=SC2016
__doc__='
Install geowatch development environment

CommandLine:
    cd $HOME/code/geowatch

    # Just show configuration
    DRY_RUN=1 ./run_developer_setup.sh

    # Execute setup with strict versions
    WATCH_STRICT=1 ./run_developer_setup.sh

Alternatives:

    export WITH_MMCV=0
    WATCH_STRICT=0 ./run_developer_setup.sh

    WATCH_STRICT=0 WITH_MMCV=1 WITH_DVC=1 WITH_COLD=1 ./run_developer_setup.sh

    # SeeAlso:
    dev/devsetup/dev_pkgs.sh
'


# Script configuration
DRY_RUN=${DRY_RUN:=0}
DEV_TRACE=${DEV_TRACE:=0}
WATCH_STRICT=${WATCH_STRICT:=0}
WITH_MMCV=${WITH_MMCV:="auto"}
WITH_TENSORFLOW=${WITH_TENSORFLOW:=0}
WITH_DVC=${WITH_DVC:=0}
WITH_AWS=${WITH_AWS:=0}
WITH_COLD=${WITH_COLD:=0}
WITH_MATERIALS=${WITH_MATERIALS:=0}
WITH_COMPAT=${WITH_COMPAT:=0}
WITH_APT_ENSURE=${WITH_APT_ENSURE:="auto"}



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
        if type sudo ; then
            sudo apt install -y "${MISS_PKGS[@]}"
        else
            apt install -y "${MISS_PKGS[@]}"
        fi
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


show_config(){
    python -c "
def identity(arg=None, *args, **kwargs):
    return arg
try:
    from ubelt import highlight_code, color_text
except ImportError:
    highlight_code = color_text = identity


print(color_text('''
=======================================
____ ____ ____ _ _ _ ____ ___ ____ _  _
| __ |___ |  | | | | |__|  |  |    |__|
|__] |___ |__| |_|_| |  |  |  |___ |  |

=======================================
''', 'green'))

print(highlight_code('''

Environment configuration:

DRY_RUN=$DRY_RUN
DEV_TRACE=$DEV_TRACE
WATCH_STRICT=$WATCH_STRICT
WITH_MMCV=$WITH_MMCV
WITH_DVC=$WITH_DVC
WITH_AWS=$WITH_AWS
WITH_COLD=$WITH_COLD
WITH_COMPAT=$WITH_COMPAT
WITH_MATERIALS=$WITH_MATERIALS
WITH_TENSORFLOW=$WITH_TENSORFLOW
WITH_APT_ENSURE=$WITH_APT_ENSURE

''', lexer_name='bash'))
    "
}


install_pytorch(){
    __doc__='
    Currently Unused.

    TODO: handle the appropriate torch version here
    Make this robust over multiple operating systems

    References:
        https://pytorch.org/
    '

    pip install ubelt parse packaging

    # Find the appropriate torch version for the devices available on this
    # machine
    TARGET_TORCH_DEVICE=$(python -c "if 1:
    from packaging.version import Version
    import ubelt as ub
    import parse

    available_cuda_versions = [
        Version('11.8'),
        Version('12.1'),
    ]

    nvcc_path = ub.find_exe('nvcc')
    if nvcc_path is None:
        print('CPU')
    else:
        parser = parse.Parser('{}, release {ver},{}')
        stdout = ub.cmd('nvcc --version').stdout
        result = parser.parse(stdout)

        cuda_version = Version(result.named['ver'])

        best = None
        for cand in available_cuda_versions:
            if cuda_version < cand:
                break
            best = cand
        print(best)
    ")
    echo "TARGET_TORCH_DEVICE = $TARGET_TORCH_DEVICE"
    if [[ "$TARGET_TORCH_DEVICE" == "cpu" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    elif [[ "$TARGET_TORCH_DEVICE" == "11.6" ]]; then
        # For CUDA 11.6, the last supported version of torch was 1.13.1
        pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    elif [[ "$TARGET_TORCH_DEVICE" == "11.8" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ "$TARGET_TORCH_DEVICE" == "12.1" ]]; then
        pip install torch torchvision torchaudio # --index-url https://download.pytorch.org/whl/cu121
    fi


}

fix_opencv_conflicts(){
    __doc__="
    Check to see if the wrong opencv is installed, and perform steps to clean
    up the incorrect libraries and install the desired (headless) ones.
    "
    # Fix opencv issues
    HAS_OPENCV_RETCODE="0"
    HAS_OPENCV_HEADLESS_RETCODE="0"
    python -m pip freeze | grep "opencv-python==" || HAS_OPENCV_RETCODE="$?"
    python -m pip freeze | grep "opencv-python-headless==" || HAS_OPENCV_HEADLESS_RETCODE="$?"

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
    __doc__="
    Unused
    "
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


check_metrics_framework(){
    __doc__="
    Check to see if the IARPA metrics framework is installed
    "
    #METRICS_MODPATH=$(python -c "import ubelt; print(ubelt.modname_to_modpath('iarpa_smart_metrics'))")
    #METRICS_MODPATH=$(python -c "import ubelt; print(ubelt.util_import._pkgutil_modname_to_modpath('iarpa_smart_metrics'))")
    METRICS_MODPATH=$(python -c "if 1:
        try:
            import iarpa_smart_metrics
        except Exception:
            print(None)
        else:
            print(iarpa_smart_metrics.__file__)
    ")
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


check_gpu_ops_work(){
    __doc__="
    quick check to ensure that GPU operations are generally functional
    "
    xdoctest -m torch --style=google --global-exec "from torch import nn\nimport torch.nn.functional as F\nimport torch" --options="+IGNORE_WHITESPACE"

    python -c "import torch; print(torch.nn.modules.Linear(10, 5).to(0)(torch.rand(10, 10).to(0)).sum().backward())"
}


main(){
    __doc__="
    The main part of the run-developer-setup script
    "
    if [[ "${DEV_TRACE}" != "0" ]]; then
        set -x
    fi

    show_config

    if [[ "$WITH_APT_ENSURE" == "auto" ]]; then
        # If on debian/ubuntu ensure the dependencies are installed
        if command_exists apt; then
            WITH_APT_ENSURE=1
        else
            WITH_APT_ENSURE=0
            echo "
            WARNING: Check and install of system packages is currently only supported
            on Debian Linux. You will need to verify that ZLIB, GSL, OpenMP are
            installed before running this script.
            "
        fi
    fi

    if [[ "$WITH_MMCV" == "auto" ]]; then
        if command_exists nvidia-smi; then
            echo "nvidia-smi detected"
            WITH_MMCV=1
        else
            echo "nvidia-smi not found"
            WITH_MMCV=0
        fi
    fi


    ###  ENSURE DEPENDENCIES ###
    if [[ "$WITH_APT_ENSURE" == "1" ]]; then
        apt_ensure ffmpeg tmux jq tree p7zip-full rsync libgsl-dev
    fi


    if [[ "$WATCH_STRICT" == "1" ]]; then
        ./dev/make_strict_req.sh
        REQUIREMENTS_DPATH=geowatch/rc/requirements-strict
    else
        REQUIREMENTS_DPATH=geowatch/rc/requirements
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
            extras.append('cold' + suffix)
        if $WITH_MATERIALS:
            extras.append('materials' + suffix)
        if $WITH_DVC:
            extras.append('dvc' + suffix)
        if $WITH_COMPAT:
            extras.append('compat' + suffix)
        print('[' + ','.join(extras) + ']')
        ")

    python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/python_build_tools.txt

    # Install the geowatch module in development mode
    python -m pip install --prefer-binary -e ".$EXTRAS"

    # Post geowatch install requirements
    python -m geowatch finish_install "--strict=$WATCH_STRICT"

    # python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/gdal.txt
    #if [[ "$WITH_COLD" == "1" ]]; then
    #    # HACK FOR COLD ISSUE
    #    #curl https://data.kitware.com/api/v1/file/6494e95df04fb36854429808/download -o pycold-0.1.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    #    #pip install "astropy==5.2.2"
    #    #pip install astropy
    #    #curl https://ipfs.io/ipfs/QmeXUmFML1BBU7jTRdvtaqbFTPBMNL9VGhvwEgrwx2wRew > pycold-311.whl
    #    #curl ipfs.io/ipfs/QmeXUmFML1BBU7jTRdvtaqbFTPBMNL9VGhvwEgrwx2wRew -o pycold-311.whl
    #    #pip install "pycold-0.1.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    #    python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/cold.txt
    #fi

    if [[ "$WITH_AWS" == "1" ]]; then
        python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/aws.txt
    fi

    if [[ "$WITH_MMCV" == "1" ]]; then

        __mmcv_notes__="
        The MMCV package is needed for DINO's deformable convolutions, and the
        correct version is specific to both your torch and cuda versions.

        The requirements/mmcv.txt only works for torch2.0 with cuda 118, so we have
        special logic here to build the correct mmcv installation command.

        To extend this logic see the mmcv website for figuring out what the correct
        string for new versions is:

        https://mmcv.readthedocs.io/en/latest/get_started/installation.html

        To test to see if your mmcv is working try running:

        .. code:: bash

            python -c 'from mmcv.ops import multi_scale_deform_attn'

        If there is no error, then it should be ok.

        Gotcha: if you have a bad mmcv, you need to uninstall it before running
        this command.  pip can't tell the difference between packages with the same
        version from different indexes.
        "

        # Logic to determine the opencv install command.
        MMCV_INSTALL_COMMAND=$(python -c "if 1:
        from packaging.version import parse as Version
        import pkg_resources
        torch_version = Version(pkg_resources.get_distribution('torch').version)

        if torch_version >= Version('2.0.0'):
            print('pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html')
        elif torch_version >= Version('1.13.0'):
            print('pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html')
        else:
            raise Exception('dont know how to deal with this version for mmcv')
        ")
        echo "MMCV_INSTALL_COMMAND = $MMCV_INSTALL_COMMAND"
        $MMCV_INSTALL_COMMAND
        #python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/mmcv.txt
    fi

    if [[ "$WITH_TENSORFLOW" == "1" ]]; then
        python -m pip install --prefer-binary -r "$REQUIREMENTS_DPATH"/tensorflow.txt
    fi

    fix_opencv_conflicts

    check_metrics_framework

    # Newer versions of torchmetrics enable pretty-errors by default, which
    # breaks tracebacks. Just uninstall it to disable this "feature".
    # https://github.com/Lightning-AI/torchmetrics/discussions/2544
    pip uninstall pretty_errors -y

    # Simple tests
    set -x
    echo "Start simple tests"
    EAGER_IMPORT_MODULES=geowatch python -c "import geowatch; print(geowatch.__version__)"
    EAGER_IMPORT_MODULES=geowatch python -m geowatch --help
    #EAGER_IMPORT=1 python -m geowatch hello_world
    python -c "import torch; print(torch.cuda.is_available())"
    set +x
}


# bpkg convention
# https://github.com/bpkg/bpkg
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    # We are sourcing the library
    show_config
    echo "Sourcing prepare_system as a library and environment"
else

    for var in "$@"
    do
        if [[ "$var" == "--help" ]]; then
            log "showing help"
            show_config
            echo "The above shows the current environment. Set the values for appropriate variables"
            echo "...exiting"
            exit 1
        fi
    done

    if [[ "$DRY_RUN" == "0" ]]; then
        # Executing file as a script
        main "${@}"
        exit $?
    else
        show_config
    fi
fi
